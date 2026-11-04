"""Try contact-guided finger closure on selected cracker-box grasps."""

import argparse
from pathlib import Path

from closure_experiment_core import Trial, add_selection_args, run_batch_subprocess, select_grasps, write_csv

from isaaclab.app import add_launcher_args


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    add_launcher_args(parser)
    add_selection_args(parser)
    parser.add_argument("--contact-threshold-n", type=float, default=0.5)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/grasp_evaluation/contact_closure"))
    parser.set_defaults(device="cpu")
    args = parser.parse_args()
    try:
        grasp_ids = select_grasps(args.grasps, args.successful_from)
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    if args.repeats < 1 or args.contact_threshold_n <= 0:
        parser.error("--repeats and --contact-threshold-n must be positive")
    if args.record_video:
        try:
            import moviepy  # noqa: F401
        except ImportError:
            parser.error("Video recording requires: uv run --extra isaacsim --extra video ...")

    trials = [
        Trial(grasp_id, repeat=repeat, stage="contact")
        for grasp_id in grasp_ids
        for repeat in range(1, args.repeats + 1)
    ]
    video_dir = (args.video_dir or args.output_dir / "videos") if args.record_video else None
    trial_file = args.output_dir / "trials.csv"
    results = []
    for offset in range(0, len(trials), 100):
        results.extend(
            run_batch_subprocess(
                trials[offset : offset + 100],
                "contact",
                args,
                video_dir=video_dir,
                contact_threshold_n=args.contact_threshold_n,
            )
        )
        write_csv(trial_file, results)
    successes = sum(bool(row["success"]) for row in results)
    print(f"Contact-guided lifts: {successes}/{len(results)}", flush=True)
    print(f"Contact-guided results saved to {trial_file}", flush=True)
    if args.record_video:
        successful_videos = [row["video_file"] for row in results if row["success"] and row.get("video_file")]
        print(f"Successful videos: {', '.join(successful_videos) or 'none in this run'}", flush=True)
    if not any(row["contact_thumb"] or row["contact_index"] or row["contact_middle"] for row in results):
        print("No stable finger-box contact was detected; inspect sensor forces in trials.csv", flush=True)


if __name__ == "__main__":
    main()
