"""Find a low fixed finger-closure percentage for selected cracker-box grasps."""

import argparse
from pathlib import Path

from closure_experiment_core import Trial, add_selection_args, run_batch_subprocess, select_grasps, write_csv

from isaaclab.app import add_launcher_args


def run_trials(trials: list[Trial], args, video_dir: Path | None = None) -> list[dict]:
    results = []
    for offset in range(0, len(trials), 100):
        results.extend(run_batch_subprocess(trials[offset : offset + 100], "fixed", args, video_dir=video_dir))
    return results


def first_coarse_success(rows: list[dict], grasp_id: int) -> int | None:
    passed = [
        int(row["closure_percent"])
        for row in rows
        if row["grasp_id"] == f"grasp_{grasp_id}" and row["stage"] == "coarse" and row["success"]
    ]
    return min(passed, default=None)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    add_launcher_args(parser)
    add_selection_args(parser)
    parser.add_argument("--closure-percent", type=int, help="Run this fixed closure directly instead of searching")
    parser.add_argument("--repeats", type=int, default=1, help="Fresh attempts at --closure-percent")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/grasp_evaluation/fixed_closure"))
    parser.set_defaults(device="cpu")
    args = parser.parse_args()
    try:
        grasp_ids = select_grasps(args.grasps, args.successful_from)
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    if args.closure_percent is not None and args.closure_percent not in range(1, 91):
        parser.error("--closure-percent must be from 1 to 90")
    if args.repeats < 1:
        parser.error("--repeats must be positive")
    if args.closure_percent is None and args.repeats != 1:
        parser.error("--repeats requires --closure-percent")
    if args.record_video:
        try:
            import moviepy  # noqa: F401
        except ImportError:
            parser.error("Video recording requires: uv run --extra isaacsim --extra video ...")

    if args.closure_percent is not None:
        trials = [
            Trial(grasp_id, args.closure_percent, repeat=repeat, stage="direct")
            for grasp_id in grasp_ids
            for repeat in range(1, args.repeats + 1)
        ]
        video_dir = (args.video_dir or args.output_dir / "videos") if args.record_video else None
        results = run_trials(trials, args, video_dir=video_dir)
        trial_file = args.output_dir / f"percent_{args.closure_percent}_trials.csv"
        write_csv(trial_file, results)
        print(
            f"Fixed {args.closure_percent}% lifts: {sum(row['success'] for row in results)}/{len(results)}", flush=True
        )
        print(f"Results saved to {trial_file}", flush=True)
        if args.record_video:
            successful_videos = [row["video_file"] for row in results if row["success"] and row.get("video_file")]
            print(f"Successful videos: {', '.join(successful_videos) or 'none in this run'}", flush=True)
        return

    trial_file = args.output_dir / "trials.csv"
    coarse = [Trial(grasp_id, percent, stage="coarse") for grasp_id in grasp_ids for percent in range(10, 91, 10)]
    print(f"Coarse search: {len(coarse)} fresh trials", flush=True)
    results = run_trials(coarse, args)
    write_csv(trial_file, results)

    refine = []
    for grasp_id in grasp_ids:
        first = first_coarse_success(results, grasp_id)
        if first is not None:
            refine.extend(Trial(grasp_id, percent, stage="refine") for percent in range(max(1, first - 9), first))
    if refine:
        print(f"One-percent refinement: {len(refine)} fresh trials", flush=True)
        results.extend(run_trials(refine, args))
        write_csv(trial_file, results)

    choices = {}
    for grasp_id in grasp_ids:
        passing = [
            int(row["closure_percent"]) for row in results if row["grasp_id"] == f"grasp_{grasp_id}" and row["success"]
        ]
        choices[grasp_id] = min(passing, default=None)

    confirmation = [
        Trial(grasp_id, percent, repeat=repeat, stage="confirm")
        for grasp_id, percent in choices.items()
        if percent is not None
        for repeat in range(1, 4)
    ]
    if confirmation:
        print(f"Confirmation: {len(confirmation)} fresh trials", flush=True)
        video_dir = (args.video_dir or args.output_dir / "videos") if args.record_video else None
        results.extend(run_trials(confirmation, args, video_dir=video_dir))
        write_csv(trial_file, results)

    summary = []
    for grasp_id in grasp_ids:
        percent = choices[grasp_id]
        confirmation_successes = sum(
            bool(row["success"])
            for row in results
            if row["grasp_id"] == f"grasp_{grasp_id}" and row["stage"] == "confirm"
        )
        summary.append(
            {
                "grasp_id": f"grasp_{grasp_id}",
                "lowest_observed_success_percent": percent if percent is not None else "",
                "confirmation_successes_of_3": confirmation_successes if percent is not None else "",
                "confirmed": confirmation_successes == 3 if percent is not None else False,
            }
        )
    choice_file = args.output_dir / "choices.csv"
    write_csv(choice_file, summary)
    print(f"Fixed-closure trials saved to {trial_file}", flush=True)
    print(f"Chosen percentages saved to {choice_file}", flush=True)
    if args.record_video and confirmation:
        print(f"Confirmation videos saved under {video_dir}", flush=True)


if __name__ == "__main__":
    main()
