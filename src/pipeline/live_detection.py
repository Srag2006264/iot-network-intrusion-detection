"""Live packet capture and N-BaIoT intrusion detection runner."""
from __future__ import annotations
import argparse
from pathlib import Path
from src.network.capture import PacketCapture
from src.pipeline.detection import DetectionPipeline
from src.prediction.random_forest_predictor import RandomForestPredictor
from src.storage.database import Database
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Live N-BaIoT IoT intrusion detection"
    )
    parser.add_argument(
        "--interface",
        default=None,
        help="Network interface to capture from",
    )
    parser.add_argument(
        "--filter",
        default="ip",
        help="BPF capture filter",
    )
    parser.add_argument(
        "--count",
        type=int,
        default=20,
        help="Number of packets to capture",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=30.0,
        help="Maximum capture time in seconds",
    )
    parser.add_argument(
        "--database",
        default="data/ids.sqlite3",
        help="SQLite database path",
    )
    args = parser.parse_args()
    # --------------------------------------------------------------
    # Database
    # --------------------------------------------------------------
    database_path = Path(args.database)
    print("=" * 70)
    print("LIVE N-BaIoT IoT INTRUSION DETECTION")
    print("=" * 70)
    print("\nSTEP 1 — INITIALIZING DATABASE")
    database = Database(database_path)
    database.initialize()
    print(f"Database: {database_path}")
    print("✓ Database initialized")
    # --------------------------------------------------------------
    # Random Forest
    # --------------------------------------------------------------
    print("\nSTEP 2 — LOADING RANDOM FOREST")
    predictor = RandomForestPredictor()
    model = predictor.load_model()
    print(f"Model: {type(model).__name__}")
    print(f"Model features: {len(model.feature_names_in_)}")
    print("✓ Random Forest loaded")
    # --------------------------------------------------------------
    # Detection pipeline
    # --------------------------------------------------------------
    print("\nSTEP 3 — CREATING DETECTION PIPELINE")
    pipeline = DetectionPipeline(
        predictor=predictor,
        database=database,
    )
    print("✓ DetectionPipeline ready")
    print(
        f"N-BaIoT features: "
        f"{pipeline.nbaiot_feature_extractor.get_num_features()}"
    )
    # --------------------------------------------------------------
    # Packet capture
    # --------------------------------------------------------------
    print("\nSTEP 4 — INITIALIZING PACKET CAPTURE")
    capture = PacketCapture(
        interface=args.interface,
        filter_expr=args.filter,
    )
    print(f"Interface: {args.interface or 'default'}")
    print(f"Filter: {args.filter}")
    print(f"Packet count: {args.count}")
    print(f"Timeout: {args.timeout}s")
    print("✓ Packet capture ready")
    # --------------------------------------------------------------
    # Live processing
    # --------------------------------------------------------------
    print("\nSTEP 5 — STARTING LIVE DETECTION")
    print("-" * 70)
    processed = 0
    normal = 0
    attacks = 0
    def process_captured_packet(packet) -> None:
        nonlocal processed
        nonlocal normal
        nonlocal attacks
        record = capture.process_packet(packet)
        if record is None:
            return
        try:
            result = pipeline.process_packet(record)
            processed += 1
            prediction = result["prediction"]
            alert = result["alert"]
            label = prediction.prediction
            probability = prediction.probability
            if label.lower() == "attack":
                attacks += 1
            else:
                normal += 1
            print(
                f"[{processed:04d}] "
                f"{record.src_ip}:{record.src_port} → "
                f"{record.dst_ip}:{record.dst_port} "
                f"{record.protocol.upper():5s} | "
                f"{label:6s} | "
                f"confidence="
                f"{probability:.3f}"
                if probability is not None
                else
                f"[{processed:04d}] "
                f"{record.src_ip}:{record.src_port} → "
                f"{record.dst_ip}:{record.dst_port} "
                f"{record.protocol.upper():5s} | "
                f"{label:6s}"
            )
            if alert is not None:
                print(
                    "       ⚠ ATTACK DETECTED — ALERT CREATED"
                )
        except Exception as exc:
            print(
                f"       ERROR processing packet: {exc}"
            )
    # --------------------------------------------------------------
    # Capture
    #
    # PacketCapture.start() already owns the Scapy sniff loop.
    # We therefore use a temporary capture callback by invoking
    # Scapy directly here.
    # --------------------------------------------------------------
    from scapy.all import sniff
    try:
        sniff(
            iface=capture.interface,
            filter=capture.filter_expr,
            prn=process_captured_packet,
            store=False,
            count=args.count,
            timeout=args.timeout,
        )
    except KeyboardInterrupt:
        print("\n\nCapture stopped by user.")
    except Exception as exc:
        print(f"\nCapture error: {exc}")
        return
    # --------------------------------------------------------------
    # Summary
    # --------------------------------------------------------------
    print("\n" + "=" * 70)
    print("LIVE DETECTION SUMMARY")
    print("=" * 70)
    print(f"Packets processed : {processed}")
    print(f"Normal predictions: {normal}")
    print(f"Attack predictions: {attacks}")
    print(f"Database          : {database_path}")
    print("=" * 70)
    if attacks > 0:
        print("⚠ ATTACK TRAFFIC DETECTED")
    else:
        print("✓ No attacks detected")
    print("=" * 70)
if __name__ == "__main__":
    main()
