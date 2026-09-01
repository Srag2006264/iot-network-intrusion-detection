"""N-BaIoT feature extraction adapter.

This module connects the project's PacketRecord representation
to the Kitsune/AfterImage network-statistics implementation.

The resulting feature vector contains 115 features:

    15 MI_dir features
    15 H features
    35 HH features
    15 HH_jit features
    35 HpHp features

Total: 115 features.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

# Kitsune was originally written for older NumPy versions where
# np.Inf existed. NumPy 2.x removed np.Inf and uses np.inf instead.
if not hasattr(np, "Inf"):
    np.Inf = np.inf  # type: ignore[attr-defined]

from src.core.contracts import PacketRecord, GenericFeatureRecord


# ---------------------------------------------------------------------------
# Locate the external Kitsune implementation
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]
KITSUNE_PATH = PROJECT_ROOT / "external" / "kitsune"

if str(KITSUNE_PATH) not in sys.path:
    sys.path.insert(0, str(KITSUNE_PATH))

import netStat  # noqa: E402


class NBAIoTFeatureExtractor:
    """Generate N-BaIoT/Kitsune-compatible network-statistics features."""

    FEATURE_VERSION = "nbaiot-kitsune-v1"

    def __init__(
        self,
        host_limit: int = 255,
        host_simplex_limit: int = 1000,
    ) -> None:
        """Initialize the Kitsune network-statistics engine."""

        self.nstat = netStat.netStat(
            np.nan,
            host_limit,
            host_simplex_limit,
        )

        self.feature_names = self._build_feature_names()

    # ------------------------------------------------------------------
    # Feature schema
    # ------------------------------------------------------------------

    def _build_feature_names(self) -> list[str]:
        """Return the exact 115-feature schema used by the trained model."""

        lambdas = ["L5", "L3", "L1", "L0.1", "L0.01"]

        names: list[str] = []

        # --------------------------------------------------------------
        # MI_dir: 5 lambdas × 3 statistics = 15
        # --------------------------------------------------------------
        for lam in lambdas:
            names.extend(
                [
                    f"MI_dir_{lam}_weight",
                    f"MI_dir_{lam}_mean",
                    f"MI_dir_{lam}_variance",
                ]
            )

        # --------------------------------------------------------------
        # H: 5 lambdas × 3 statistics = 15
        # --------------------------------------------------------------
        for lam in lambdas:
            names.extend(
                [
                    f"H_{lam}_weight",
                    f"H_{lam}_mean",
                    f"H_{lam}_variance",
                ]
            )

        # --------------------------------------------------------------
        # HH: 5 lambdas × 7 statistics = 35
        # --------------------------------------------------------------
        for lam in lambdas:
            names.extend(
                [
                    f"HH_{lam}_weight",
                    f"HH_{lam}_mean",
                    f"HH_{lam}_std",
                    f"HH_{lam}_magnitude",
                    f"HH_{lam}_radius",
                    f"HH_{lam}_covariance",
                    f"HH_{lam}_pcc",
                ]
            )

        # --------------------------------------------------------------
        # HH_jit: 5 lambdas × 3 statistics = 15
        # --------------------------------------------------------------
        for lam in lambdas:
            names.extend(
                [
                    f"HH_jit_{lam}_weight",
                    f"HH_jit_{lam}_mean",
                    f"HH_jit_{lam}_variance",
                ]
            )

        # --------------------------------------------------------------
        # HpHp: 5 lambdas × 7 statistics = 35
        # --------------------------------------------------------------
        for lam in lambdas:
            names.extend(
                [
                    f"HpHp_{lam}_weight",
                    f"HpHp_{lam}_mean",
                    f"HpHp_{lam}_std",
                    f"HpHp_{lam}_magnitude",
                    f"HpHp_{lam}_radius",
                    f"HpHp_{lam}_covariance",
                    f"HpHp_{lam}_pcc",
                ]
            )

        if len(names) != 115:
            raise ValueError(
                f"Expected 115 feature names, got {len(names)}"
            )

        return names

    # ------------------------------------------------------------------
    # Packet conversion
    # ------------------------------------------------------------------

    @staticmethod
    def _packet_protocol(packet: PacketRecord) -> tuple[str, str]:
        """Convert PacketRecord protocol/ports to Kitsune identifiers."""

        protocol = packet.protocol.lower()

        if protocol == "tcp":
            return str(packet.src_port or ""), str(packet.dst_port or "")

        if protocol == "udp":
            return str(packet.src_port or ""), str(packet.dst_port or "")

        if protocol == "arp":
            return "arp", "arp"

        if protocol == "icmp":
            return "icmp", "icmp"

        return "", ""

    @staticmethod
    def _ip_type(packet: PacketRecord) -> int:
        """Return Kitsune IP type.

        0 = IPv4
        1 = IPv6
        """

        if ":" in packet.src_ip or ":" in packet.dst_ip:
            return 1

        return 0

    # ------------------------------------------------------------------
    # Feature extraction
    # ------------------------------------------------------------------

    def extract(self, packet: PacketRecord) -> dict[str, float]:
        """Process one packet and return the current 115 features."""

        if not isinstance(packet, PacketRecord):
            raise TypeError("packet must be a PacketRecord instance")

        src_protocol, dst_protocol = self._packet_protocol(packet)

        ip_type = self._ip_type(packet)

        # PacketRecord currently does not carry MAC addresses.
        # Empty strings are therefore used for the Kitsune MAC fields.
        src_mac = ""
        dst_mac = ""

        # --------------------------------------------------------------
        # Original Kitsune statistics
        # --------------------------------------------------------------

        kitsune_features = self.nstat.updateGetStats(
            ip_type,
            src_mac,
            dst_mac,
            packet.src_ip,
            src_protocol,
            packet.dst_ip,
            dst_protocol,
            int(packet.packet_length),
            float(packet.timestamp),
        )

        kitsune_features = np.asarray(
            kitsune_features,
            dtype=float,
        )

        if len(kitsune_features) != 100:
            raise ValueError(
                f"Expected 100 Kitsune features, "
                f"got {len(kitsune_features)}"
            )

        # --------------------------------------------------------------
        # Add the missing H features
        # --------------------------------------------------------------

        h_features: list[float] = []

        for lam in [5, 3, 1, 0.1, 0.01]:
            stats = self.nstat.HT_H.update_get_1D_Stats(
                packet.src_ip,
                float(packet.timestamp),
                int(packet.packet_length),
                lam,
            )

            h_features.extend(
                [
                    float(stats[0]),
                    float(stats[1]),
                    float(stats[2]),
                ]
            )

        if len(h_features) != 15:
            raise ValueError(
                f"Expected 15 H features, got {len(h_features)}"
            )

        # --------------------------------------------------------------
        # Kitsune returns:
        #
        # MI_dir + HH + HH_jit + HpHp
        #
        # We need:
        #
        # MI_dir + H + HH + HH_jit + HpHp
        # --------------------------------------------------------------

        mi_dir = kitsune_features[0:15]
        hh = kitsune_features[15:50]
        hh_jit = kitsune_features[50:65]
        hphp = kitsune_features[65:100]

        features = np.concatenate(
            [
                mi_dir,
                np.asarray(h_features),
                hh,
                hh_jit,
                hphp,
            ]
        )

        if len(features) != 115:
            raise ValueError(
                f"Expected 115 N-BaIoT features, got {len(features)}"
            )

        return {
            name: float(value)
            for name, value in zip(
                self.feature_names,
                features,
            )
        }

    # ------------------------------------------------------------------
    # GenericFeatureRecord adapter
    # ------------------------------------------------------------------

    def extract_record(
        self,
        packet: PacketRecord,
    ) -> GenericFeatureRecord:
        """Extract N-BaIoT features as a GenericFeatureRecord."""

        features = self.extract(packet)

        return GenericFeatureRecord(
            feature_version=self.FEATURE_VERSION,
            features=features,
            metadata={
                "timestamp": float(packet.timestamp),
                "src_ip": packet.src_ip,
                "dst_ip": packet.dst_ip,
                "src_port": packet.src_port,
                "dst_port": packet.dst_port,
                "protocol": packet.protocol,
                "feature_source": "nbaiot_kitsune_feature_extractor",
            },
        )

    # ------------------------------------------------------------------
    # Ordered feature vector
    # ------------------------------------------------------------------

    def extract_vector(
        self,
        packet: PacketRecord,
    ) -> list[float]:
        """Return the 115 features as an ordered list."""

        feature_dict = self.extract(packet)

        return [
            feature_dict[name]
            for name in self.feature_names
        ]

    # ------------------------------------------------------------------
    # Schema information
    # ------------------------------------------------------------------

    def get_feature_names(self) -> list[str]:
        """Return the ordered N-BaIoT feature names."""

        return list(self.feature_names)

    def get_num_features(self) -> int:
        """Return the number of generated features."""

        return len(self.feature_names)