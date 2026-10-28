"""Command-line entry points for dataset selection and auditing."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import typer
from rich.console import Console
from rich.table import Table

from .audit import audit_sequence, project_rotations_to_so3, write_audit
from .contact import PHASE_NAMES, analyze_join_sequence
from .core4d import infer_object_category, load_action_labels, select_sequences
from .remote_archive import (
    V1_MOTIONS_URL,
    V1_SEQUENCE_FILENAMES,
    V2_BATCH2_URL,
    V2_SEQUENCE_MEMBERS,
    extract_sequence,
    extract_sequences,
    read_sequence_metadata,
)
from .visualization import (
    plot_contact_timeline,
    plot_metric_comparison,
    render_independent_retarget_gif,
    render_source_sequence_gif,
)

app = typer.Typer(no_args_is_help=True)
console = Console()


@app.command("select")
def select(
    labels_path: Path = typer.Option(
        Path("data/core4d/metadata/action_labels.json"),
        exists=True,
        dir_okay=False,
        help="Path to the official action_labels.json.",
    ),
    action: str = typer.Option("join", help="Normalized action name."),
    obstacle: str = typer.Option("obs0", help="Obstacle label."),
    output: Path | None = typer.Option(None, help="Optional JSON output manifest."),
) -> None:
    """List CORE4D sequence IDs matching an action/obstacle query."""
    matches = select_sequences(
        load_action_labels(labels_path),
        action=action,
        obstacle=obstacle,
    )

    table = Table(title=f"CORE4D candidates: {action} + {obstacle}")
    table.add_column("#", justify="right")
    table.add_column("Sequence ID")
    for index, match in enumerate(matches, start=1):
        table.add_row(str(index), match.sequence_id)
    console.print(table)
    console.print(f"Total: {len(matches)}")

    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "query": {"action": action, "obstacle": obstacle},
            "count": len(matches),
            "sequence_ids": [match.sequence_id for match in matches],
        }
        output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        console.print(f"Wrote {output}")


@app.command("fetch-object-metadata")
def fetch_object_metadata(
    labels_path: Path = typer.Option(
        Path("data/core4d/metadata/action_labels.json"),
        exists=True,
        dir_okay=False,
    ),
    archive_url: str = typer.Option(V1_MOTIONS_URL, help="Remote CORE4D V1 ZIP URL."),
    output: Path = typer.Option(
        Path("data/core4d/metadata/join_obs0_objects.json"),
        help="Output metadata manifest.",
    ),
) -> None:
    """Fetch tiny object metadata files for all join+obs0 candidates."""
    matches = select_sequences(
        load_action_labels(labels_path),
        action="join",
        obstacle="obs0",
    )
    metadata = read_sequence_metadata(
        archive_url,
        (match.sequence_id for match in matches),
    )
    payload = {
        sequence_id: {
            "category": infer_object_category(value),
            "metadata": value,
        }
        for sequence_id, value in sorted(metadata.items())
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    box_count = sum(item["category"] == "box" for item in payload.values())
    console.print(f"Fetched metadata for {len(payload)} sequences")
    console.print(f"Detected box sequences: {box_count}")
    console.print(f"Wrote {output}")


@app.command("fetch-sequence")
def fetch_sequence(
    sequence_id: str = typer.Argument(help="CORE4D sequence ID, e.g. 20231108/000."),
    v1_archive_url: str = typer.Option(
        V1_MOTIONS_URL,
        help="Remote CORE4D V1 motion ZIP URL.",
    ),
    v2_archive_url: str = typer.Option(
        V2_BATCH2_URL,
        help="Remote CORE4D V2 human-motion ZIP URL.",
    ),
    destination: Path = typer.Option(
        Path("data/core4d/motions"),
        help="Local motion root.",
    ),
) -> None:
    """Merge V1 object data and V2 human fits without full archive downloads."""
    written = extract_sequence(v1_archive_url, sequence_id, destination)
    written.extend(
        extract_sequence(
            v2_archive_url,
            sequence_id,
            destination,
            filenames=V2_SEQUENCE_MEMBERS,
            rename=V2_SEQUENCE_MEMBERS,
        )
    )
    if not written:
        raise typer.Exit(code=1)
    for path in written:
        console.print(path)


@app.command("fetch-box-sequences")
def fetch_box_sequences(
    manifest_path: Path = typer.Option(
        Path("data/core4d/metadata/join_obs0_objects.json"),
        exists=True,
        dir_okay=False,
    ),
    destination: Path = typer.Option(Path("data/core4d/motions")),
) -> None:
    """Selectively fetch all six join+box+obs0 sequences from V1 and V2."""
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    sequence_ids = sorted(
        sequence_id
        for sequence_id, value in manifest.items()
        if value.get("category") == "box"
    )
    if not sequence_ids:
        raise typer.BadParameter("Manifest contains no box sequences")
    v1_results = extract_sequences(
        V1_MOTIONS_URL,
        sequence_ids,
        destination,
        V1_SEQUENCE_FILENAMES,
    )
    v2_results = extract_sequences(
        V2_BATCH2_URL,
        sequence_ids,
        destination,
        V2_SEQUENCE_MEMBERS,
        rename=V2_SEQUENCE_MEMBERS,
    )
    for sequence_id in sequence_ids:
        files = v1_results[sequence_id] + v2_results[sequence_id]
        console.print(f"{sequence_id}: {len(files)} files")


@app.command("audit")
def audit(
    sequence_id: str = typer.Argument(help="CORE4D sequence ID."),
    motion_root: Path = typer.Option(
        Path("data/core4d/motions"),
        exists=True,
        file_okay=False,
    ),
    output_root: Path = typer.Option(Path("outputs/audits")),
) -> None:
    """Validate one local CORE4D sequence and write a JSON report."""
    report = audit_sequence(motion_root, sequence_id)
    output = output_root / f"{sequence_id.replace('/', '_')}.json"
    write_audit(report, output)
    console.print(f"valid={report.valid}")
    console.print(f"frames={report.frame_counts}")
    for warning in report.warnings:
        console.print(f"[yellow]{warning}[/yellow]")
    for error in report.errors:
        console.print(f"[red]{error}[/red]")
    console.print(f"Wrote {output}")
    if not report.valid:
        raise typer.Exit(code=1)


@app.command("repair-object-poses")
def repair_object_poses(
    input_path: Path = typer.Argument(exists=True, dir_okay=False),
    output_path: Path = typer.Argument(),
) -> None:
    """Project noisy object rotations to SO(3), preserving the source file."""
    poses = np.load(input_path)
    repaired = project_rotations_to_so3(poses)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(output_path, repaired)
    max_change = float(np.abs(repaired - poses).max())
    console.print(f"frames={poses.shape[0]}")
    console.print(f"max_element_change={max_change:.6g}")
    console.print(f"Wrote {output_path}")


@app.command("analyze-join")
def analyze_join(
    sequence_id: str = typer.Argument(help="CORE4D join sequence ID."),
    motion_root: Path = typer.Option(Path("data/core4d/motions")),
    object_models_root: Path = typer.Option(Path("data/core4d/object_models")),
    output_root: Path = typer.Option(Path("outputs/contact_analysis")),
) -> None:
    """Detect object contacts and four temporal phases of a join sequence."""
    analysis, arrays = analyze_join_sequence(motion_root, sequence_id, object_models_root)
    stem = sequence_id.replace("/", "_")
    output_root.mkdir(parents=True, exist_ok=True)
    json_path = output_root / f"{stem}.json"
    npz_path = output_root / f"{stem}.npz"
    payload = asdict(analysis)
    payload["phase_names"] = PHASE_NAMES
    json_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    np.savez_compressed(npz_path, **arrays)
    console.print(json.dumps(payload, indent=2))
    console.print(f"Wrote {json_path}")
    console.print(f"Wrote {npz_path}")


@app.command("plot-contact")
def plot_contact(
    analysis_npz: Path = typer.Argument(exists=True, dir_okay=False),
    output_path: Path = typer.Argument(),
) -> None:
    """Render the contact distances and detected join phases."""
    plot_contact_timeline(analysis_npz, output_path)
    console.print(f"Wrote {output_path}")


@app.command("render-source")
def render_source(
    sequence_id: str = typer.Argument(help="CORE4D sequence ID."),
    motion_root: Path = typer.Option(Path("data/core4d/motions")),
    object_models_root: Path = typer.Option(Path("data/core4d/object_models")),
    output_path: Path = typer.Option(
        Path("outputs/source_visualization/preview.gif"),
    ),
    frame_step: int = typer.Option(2, min=1),
) -> None:
    """Render a skeleton/object GIF without requiring licensed SMPL-X models."""
    render_source_sequence_gif(
        motion_root,
        sequence_id,
        object_models_root,
        output_path,
        frame_step=frame_step,
    )
    console.print(f"Wrote {output_path}")


@app.command("retarget-independent")
def retarget_independent_command(
    sequence_id: str = typer.Argument(help="CORE4D sequence ID."),
    person: int = typer.Option(..., min=1, max=2),
    motion_root: Path = typer.Option(Path("data/core4d/motions")),
    urdf_path: Path = typer.Option(
        Path("third_party/unitree_ros/robots/g1_description/g1_29dof_rev_1_0.urdf"),
        exists=True,
        dir_okay=False,
    ),
    output_root: Path = typer.Option(Path("outputs/independent")),
    iterations: int = typer.Option(400, min=1),
) -> None:
    """Fit one human independently to the official 29-DoF G1 model."""
    from .audit import load_npz_mapping
    from .retarget.independent import (
        IndependentRetargetConfig,
        retarget_independent,
        save_independent_result,
    )

    sequence_dir = motion_root.joinpath(*sequence_id.split("/"))
    source = load_npz_mapping(sequence_dir / f"person{person}_poses.npz")
    result = retarget_independent(
        np.asarray(source["joints"]),
        urdf_path,
        IndependentRetargetConfig(iterations=iterations),
    )
    output_path = output_root / sequence_id.replace("/", "_") / f"robot{person}.npz"
    save_independent_result(result, output_path)
    error_cm = float(
        np.linalg.norm(result.robot_keypoints - result.target_keypoints, axis=-1).mean() * 100
    )
    console.print(f"frames={len(result.joint_positions)}")
    console.print(f"scale={result.scale:.4f}")
    console.print(f"mean_semantic_keypoint_error_cm={error_cm:.3f}")
    console.print(f"initial_loss={result.loss_history[0]:.6f}")
    console.print(f"final_loss={result.loss_history[-1]:.6f}")
    console.print(f"Wrote {output_path}")


@app.command("evaluate-independent")
def evaluate_independent_command(
    sequence_id: str = typer.Argument(help="CORE4D sequence ID."),
    motion_root: Path = typer.Option(Path("data/core4d/motions")),
    object_models_root: Path = typer.Option(Path("data/core4d/object_models")),
    contact_analysis_root: Path = typer.Option(Path("outputs/contact_analysis")),
    result_root: Path = typer.Option(Path("outputs/independent")),
    output_root: Path = typer.Option(Path("outputs/metrics/independent")),
) -> None:
    """Evaluate both independent G1 trajectories against CORE4D contacts."""
    from .metrics import evaluate_independent_sequence, write_metrics

    stem = sequence_id.replace("/", "_")
    metrics = evaluate_independent_sequence(
        motion_root,
        sequence_id,
        object_models_root,
        contact_analysis_root / f"{stem}.npz",
        result_root,
    )
    output_path = output_root / f"{stem}.json"
    write_metrics(metrics, output_path)
    console.print(json.dumps(metrics, indent=2))
    console.print(f"Wrote {output_path}")


@app.command("evaluate-paper-join")
def evaluate_paper_join_command(
    sequence_id: str = typer.Argument(help="CORE4D join sequence ID."),
    motion_root: Path = typer.Option(Path("data/core4d/motions")),
    object_models_root: Path = typer.Option(Path("data/core4d/object_models")),
    contact_analysis_root: Path = typer.Option(Path("outputs/contact_analysis")),
    result_root: Path = typer.Option(..., help="Root containing <stem>/robot*.npz."),
    output_path: Path = typer.Option(..., help="Paper metrics JSON path."),
    reference_root: Path = typer.Option(Path("outputs/independent")),
    contact_enter: float = typer.Option(0.04),
    contact_exit: float = typer.Option(0.055),
) -> None:
    """Evaluate one result with config-neutral paper metric definitions."""
    from .metrics import evaluate_independent_sequence, write_metrics

    stem = sequence_id.replace("/", "_")
    metrics = evaluate_independent_sequence(
        motion_root,
        sequence_id,
        object_models_root,
        contact_analysis_root / f"{stem}.npz",
        result_root,
        robot_contact_enter=contact_enter,
        robot_contact_exit=contact_exit,
        paper_metric_mode=True,
        reference_root=reference_root,
    )
    write_metrics(metrics, output_path)
    console.print(json.dumps(metrics, indent=2))
    console.print(f"Wrote {output_path}")


@app.command("report-geometry")
def report_geometry_command(
    metrics_root: Path = typer.Option(Path("outputs/metrics/balanced_clearance_v10_smooth")),
    output_path: Path = typer.Option(Path("outputs/metrics/geometry_v10.json")),
) -> None:
    """Write the literature geometry table from evaluate-independent JSON files."""
    from .metrics import write_geometry_report

    report = write_geometry_report(metrics_root, output_path)
    console.print(json.dumps(report, indent=2))
    console.print(f"Wrote {output_path}")


@app.command("summarize-paper-join")
def summarize_paper_join_command(
    paper_root: Path = typer.Option(Path("outputs/paper_eval_join")),
) -> None:
    """Write paper CSV/Markdown/LaTeX tables and failure notes."""
    from .dataset import BOX_SEQUENCES
    from .paper_eval import (
        ablation_rows,
        load_paper_rows,
        summarize_sequence_rows,
        write_ablation_component_matrix,
        write_ablation_tables,
        write_aggregate_markdown,
        write_csv,
        write_failure_cases,
    )
    from .retarget.paper_presets import PAPER_VARIANTS

    final_paths = [
        paper_root / "final_six" / sequence_id.rsplit("/", 1)[-1] / "metrics.json"
        for sequence_id in BOX_SEQUENCES
    ]
    missing = [path for path in final_paths if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing final-six metrics: {missing}")
    rows = load_paper_rows(final_paths)
    summary_root = paper_root / "summary"
    write_csv(rows, summary_root / "join_sequence_summary.csv")
    aggregate = summarize_sequence_rows(rows)
    (summary_root / "join_six_sequence_aggregate.json").write_text(
        json.dumps(aggregate, indent=2) + "\n", encoding="utf-8"
    )
    write_aggregate_markdown(
        aggregate, summary_root / "join_six_sequence_aggregate.md"
    )
    write_failure_cases(rows, summary_root / "failure_cases.md")

    ablation_paths = {
        variant: paper_root / "ablation_018" / variant / "metrics.json"
        for variant in PAPER_VARIANTS
    }
    missing = [path for path in ablation_paths.values() if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing ablation metrics: {missing}")
    write_ablation_tables(ablation_rows(ablation_paths), summary_root)
    write_ablation_component_matrix(summary_root)
    console.print(f"Wrote paper summaries under {summary_root}")


@app.command("copy-paper-join-final")
def copy_paper_join_final_command(
    frozen_root: Path = typer.Option(Path("outputs/balanced_clearance_v41_smooth")),
    frozen_metrics_root: Path = typer.Option(
        Path("outputs/metrics/balanced_clearance_v41_smooth")
    ),
    frozen_video_root: Path = typer.Option(Path("videos/retarget")),
    output_root: Path = typer.Option(Path("outputs/paper_eval_join/final_six")),
) -> None:
    """Copy the frozen v41 six-sequence artifacts without modifying them."""
    from .dataset import BOX_SEQUENCES
    from .paper_eval import copy_frozen_sequence

    for sequence_id in BOX_SEQUENCES:
        destination = copy_frozen_sequence(
            sequence_id,
            frozen_root,
            frozen_metrics_root,
            frozen_video_root,
            output_root,
        )
        console.print(f"Wrote {destination}")


@app.command("plot-paper-join")
def plot_paper_join_command(
    paper_root: Path = typer.Option(Path("outputs/paper_eval_join")),
    sequence_id: str = typer.Option("20231108/018"),
    motion_root: Path = typer.Option(Path("data/core4d/motions")),
    object_models_root: Path = typer.Option(Path("data/core4d/object_models")),
    contact_analysis_root: Path = typer.Option(Path("outputs/contact_analysis")),
    source_media: Path = typer.Option(
        Path("outputs/paper_eval_join/figures/source_018.gif")
    ),
) -> None:
    """Generate final-result main figures and controlled-ablation supplements."""
    from .paper_figures import (
        plot_join_phase_contact_timeline,
        plot_join_qualitative_comparison,
    )

    figures = paper_root / "figures"
    ablation = paper_root / "ablation_018"
    frozen = paper_root / "final_six" / "018"

    # Main-paper qualitative figure: source, independent baseline, accepted final.
    plot_join_qualitative_comparison(
        source_media,
        {
            "Independent IK": ablation / "J0" / "preview.mp4",
            "Frozen v41": frozen / "frozen_preview.mp4",
        },
        figures / "join_final_qualitative_comparison.png",
    )

    # Supplemental controlled ablation remains available separately.
    plot_join_qualitative_comparison(
        source_media,
        {
            "J0 Independent": ablation / "J0" / "preview.mp4",
            "J2 Phase-aware": ablation / "J2" / "preview.mp4",
            "J4 Full": ablation / "J4" / "preview.mp4",
        },
        figures / "join_ablation_qualitative_comparison.png",
    )
    stem = sequence_id.replace("/", "_")

    # Main-paper timeline uses the accepted warm-started v41 result.
    plot_join_phase_contact_timeline(
        motion_root,
        sequence_id,
        object_models_root,
        contact_analysis_root / f"{stem}.npz",
        frozen,
        figures / "join_frozen_v41_contact_timeline.png",
        method_label="Frozen v41 (warm-started final)",
    )

    # Supplemental timeline documents the controlled cold-start J4 behavior.
    plot_join_phase_contact_timeline(
        motion_root,
        sequence_id,
        object_models_root,
        contact_analysis_root / f"{stem}.npz",
        ablation / "J4" / "smooth",
        figures / "join_ablation_J4_contact_timeline.png",
        method_label="Cold-start J4 (controlled ablation)",
    )
    console.print(f"Wrote paper figures under {figures}")


@app.command("freeze-paper-join")
def freeze_paper_join_command(
    repo_root: Path = typer.Option(Path(".")),
    paper_root: Path = typer.Option(Path("outputs/paper_eval_join")),
    output_root: Path = typer.Option(Path("outputs/paper_release_join")),
) -> None:
    """Freeze the evaluated Join paper artifacts with provenance/checksums."""
    from .paper_release import freeze_join_paper_release

    freeze_join_paper_release(repo_root.resolve(), paper_root, output_root)
    console.print(f"Frozen paper release at {output_root}")


@app.command("evaluate-coordination")
def evaluate_coordination_command(
    sequence_id: str = typer.Argument(help="CORE4D join sequence ID."),
    result_root: Path = typer.Option(..., "--result-root"),
    label: str = typer.Option("result", "--label"),
    output_path: Path = typer.Option(..., "--output-path"),
    motion_root: Path = typer.Option(Path("data/core4d/motions")),
    object_models_root: Path = typer.Option(Path("data/core4d/object_models")),
    contact_analysis_root: Path = typer.Option(Path("outputs/contact_analysis")),
) -> None:
    """Post-hoc cross-agent coordination metrics for one saved dual-robot result."""
    from .coordination_report import evaluate_single

    metrics = evaluate_single(
        result_root=result_root,
        sequence_id=sequence_id,
        label=label,
        output_path=output_path,
        motion_root=motion_root,
        object_models_root=object_models_root,
        contact_analysis_root=contact_analysis_root,
    )
    jc = metrics["joint_contact"]
    console.print(
        f"joint F1={jc['f1']:.3f} IoU={jc['iou']:.3f} coverage={jc['recall']:.3f}"
    )
    console.print(f"Wrote {output_path}")


@app.command("coordination-extension")
def coordination_extension_command(
    output_root: Path = typer.Option(Path("outputs/paper_eval/coordination_extension")),
    paper_root: Path = typer.Option(Path("outputs/paper_eval_join")),
    coupling_root: Path | None = typer.Option(None, "--coupling-root"),
    frozen_root: Path = typer.Option(Path("outputs/balanced_clearance_v41_smooth")),
    motion_root: Path = typer.Option(Path("data/core4d/motions")),
    object_models_root: Path = typer.Option(Path("data/core4d/object_models")),
    contact_analysis_root: Path = typer.Option(Path("outputs/contact_analysis")),
) -> None:
    """Evaluate J0--J4, staged v41 (six takes), and coupling controls post hoc."""
    from .coordination_report import run_coordination_extension

    provenance = run_coordination_extension(
        output_root=output_root,
        paper_root=paper_root,
        coupling_root=coupling_root,
        frozen_root=frozen_root,
        motion_root=motion_root,
        object_models_root=object_models_root,
        contact_analysis_root=contact_analysis_root,
    )
    console.print(
        f"Evaluated {provenance['n_results']} results; frozen copies identical="
        f"{provenance['all_frozen_copies_identical']}"
    )
    console.print(f"Wrote coordination extension under {output_root}")


@app.command("render-independent")
def render_independent_command(
    sequence_id: str = typer.Argument(help="CORE4D sequence ID."),
    motion_root: Path = typer.Option(Path("data/core4d/motions")),
    object_models_root: Path = typer.Option(Path("data/core4d/object_models")),
    result_root: Path = typer.Option(Path("outputs/independent")),
    output_path: Path = typer.Option(
        Path("outputs/independent/preview.gif"),
    ),
    frame_step: int = typer.Option(2, min=1),
    method_label: str = typer.Option("Independent baseline"),
) -> None:
    """Render the two independent G1 baselines around the shared object."""
    render_independent_retarget_gif(
        motion_root,
        sequence_id,
        object_models_root,
        result_root,
        output_path,
        frame_step=frame_step,
        method_label=method_label,
    )
    console.print(f"Wrote {output_path}")


@app.command("retarget-coupled")
def retarget_coupled_command(
    sequence_id: str = typer.Argument(help="CORE4D sequence ID."),
    motion_root: Path = typer.Option(Path("data/core4d/motions")),
    object_models_root: Path = typer.Option(Path("data/core4d/object_models")),
    baseline_root: Path = typer.Option(Path("outputs/independent")),
    contact_analysis_root: Path = typer.Option(Path("outputs/contact_analysis")),
    urdf_path: Path = typer.Option(
        Path("third_party/unitree_ros/robots/g1_description/g1_29dof_rev_1_0.urdf"),
        exists=True,
        dir_okay=False,
    ),
    output_root: Path = typer.Option(Path("outputs/coupled")),
    iterations: int = typer.Option(600, min=1),
    balance: bool = typer.Option(False, "--balance"),
    clearance: bool = typer.Option(False, "--clearance"),
    hull_orient: bool = typer.Option(
        False,
        "--hull-orient",
        help="With --clearance: wrist/ankle rotation-only hull (v17). Does not change the v13 default.",
    ),
    hull_distal: bool = typer.Option(
        False,
        "--hull-distal",
        help="With --clearance: fingertip-half hull from v13 (v18). Does not change the v13 default.",
    ),
    from_v13: bool = typer.Option(
        False,
        "--from-v13",
        help="Initialize from outputs/balanced_clearance_v13_smooth instead of independent.",
    ),
    from_v20: bool = typer.Option(
        False,
        "--from-v20",
        help="Initialize from unsmoothed outputs/balanced_clearance_v20.",
    ),
    from_v23: bool = typer.Option(
        False,
        "--from-v23",
        help="Initialize from outputs/balanced_clearance_v23_smooth.",
    ),
    from_v24: bool = typer.Option(
        False,
        "--from-v24",
        help="Initialize from outputs/balanced_clearance_v24_smooth.",
    ),
    from_v30: bool = typer.Option(
        False,
        "--from-v30",
        help="Initialize from outputs/balanced_clearance_v30_smooth.",
    ),
    from_v31: bool = typer.Option(
        False,
        "--from-v31",
        help="Initialize from outputs/balanced_clearance_v31_smooth.",
    ),
    from_v32: bool = typer.Option(
        False,
        "--from-v32",
        help="Initialize from outputs/balanced_clearance_v32_smooth.",
    ),
    from_v33: bool = typer.Option(
        False,
        "--from-v33",
        help="Initialize from outputs/balanced_clearance_v33_smooth.",
    ),
    from_v34: bool = typer.Option(
        False,
        "--from-v34",
        help="Initialize from outputs/balanced_clearance_v34_smooth.",
    ),
    from_v35: bool = typer.Option(
        False,
        "--from-v35",
        help="Initialize from outputs/balanced_clearance_v35_smooth.",
    ),
    from_v36: bool = typer.Option(
        False,
        "--from-v36",
        help="Initialize from outputs/balanced_clearance_v36_smooth.",
    ),
    from_v39: bool = typer.Option(
        False,
        "--from-v39",
        help="Initialize from outputs/balanced_clearance_v39_smooth.",
    ),
    from_v40: bool = typer.Option(
        False,
        "--from-v40",
        help="Initialize from outputs/balanced_clearance_v40_smooth.",
    ),
    mesh_hull: bool = typer.Option(
        False,
        "--mesh-hull",
        help="With --clearance: slack hull on STL samples (v21). Writes a new folder.",
    ),
    omni_geom: bool = typer.Option(
        False,
        "--omni-geom",
        help="With --clearance: idle dual-plant on the join (v41). 018-first.",
    ),
) -> None:
    """Jointly refine both G1 trajectories using shared-object contacts."""
    from .retarget.coupled import (
        CoupledRetargetConfig,
        retarget_coupled,
        save_coupled_results,
    )

    stem = sequence_id.replace("/", "_")
    config_kwargs: dict[str, float | int | bool] = {"iterations": iterations}
    if balance or clearance:
        config_kwargs.update(
            {
                "com_support_weight": 40.0,
                "foot_height_weight": 20.0,
                "foot_slip_weight": 30.0,
                "torso_upright_weight": 8.0,
                "com_center_weight": 20.0,
                "jerk_weight": 25.0,
                "acceleration_weight": 16.0,
            }
        )
        if output_root == Path("outputs/coupled"):
            output_root = Path("outputs/balanced")
    if clearance:
        config_kwargs.update(
            {
                "object_clearance_weight": 80.0,
                "object_clearance_margin_m": 0.14,
                "hand_clearance_margin_m": 0.0,
                "hand_surface_clearance_margin_m": 0.0,
                "hand_volume_clearance_weight": 400.0,
                "hand_volume_clearance_margin_m": 0.0,
                "torso_volume_clearance_weight": 80.0,
                "torso_volume_clearance_margin_m": 0.04,
                "palm_offset_m": 0.0,
                "contact_weight": 90.0,
                "pose_fidelity_weight": 12.0,
                "planted_root_slip_weight": 400.0,
                "foot_slip_weight": 400.0,
                "foot_volume_clearance_weight": 250.0,
                "foot_volume_clearance_margin_m": 0.02,
                "foot_ground_weight": 400.0,
                "foot_slide_weight": 400.0,
                "foot_height_weight": 50.0,
                "hard_plant_weight": 800.0,
                "plant_height_m": 0.02,
                "source_slow_xy_m_s": 0.05,
                "plant_require_contact": False,
                "object_hull_clearance_weight": 0.0,
                "jerk_weight": 32.0,
                "inter_robot_weight": 8.0,
                "inter_robot_clearance": 0.12,
            }
        )
        if output_root in (
            Path("outputs/coupled"),
            Path("outputs/balanced"),
        ):
            output_root = Path("outputs/balanced_clearance_v20")
        if hull_orient and not hull_distal:
            config_kwargs["object_hull_clearance_weight"] = 800.0
            config_kwargs["hull_orient_only"] = True
            if output_root in (
                Path("outputs/coupled"),
                Path("outputs/balanced"),
                Path("outputs/balanced_clearance_v13"),
                Path("outputs/balanced_clearance_v20"),
            ):
                output_root = Path("outputs/balanced_clearance_v17")
        if hull_distal:
            config_kwargs["object_hull_clearance_weight"] = 800.0 if hull_orient else 400.0
            config_kwargs["hull_orient_only"] = bool(hull_orient)
            config_kwargs["hull_min_local_x_m"] = 0.09
            config_kwargs["hull_mask_aabb"] = False
            config_kwargs["root_fidelity_weight"] = 60.0
            config_kwargs["pose_fidelity_weight"] = 18.0
            config_kwargs["learning_rate"] = 0.008
            tag = "v19" if hull_orient else "v18"
            if output_root in (
                Path("outputs/coupled"),
                Path("outputs/balanced"),
                Path("outputs/balanced_clearance_v13"),
                Path("outputs/balanced_clearance_v17"),
                Path("outputs/balanced_clearance_v18"),
                Path("outputs/balanced_clearance_v20"),
            ):
                output_root = Path(f"outputs/balanced_clearance_{tag}")
        if mesh_hull:
            config_kwargs["object_hull_clearance_weight"] = 80.0
            config_kwargs["hull_orient_only"] = False
            config_kwargs["hull_mask_aabb"] = False
            config_kwargs["hull_use_contact_mask"] = False
            config_kwargs["hull_slack_m"] = 0.008
            config_kwargs["contact_weight"] = 90.0
            config_kwargs["jerk_weight"] = 48.0
            if from_v20:
                config_kwargs["root_fidelity_weight"] = 50.0
                config_kwargs["pose_fidelity_weight"] = 16.0
                config_kwargs["learning_rate"] = 0.01
            if output_root in (
                Path("outputs/coupled"),
                Path("outputs/balanced"),
                Path("outputs/balanced_clearance_v13"),
                Path("outputs/balanced_clearance_v17"),
                Path("outputs/balanced_clearance_v18"),
                Path("outputs/balanced_clearance_v19"),
                Path("outputs/balanced_clearance_v20"),
            ):
                output_root = Path("outputs/balanced_clearance_v21")
        if omni_geom:
            config_kwargs["plant_require_contact"] = True
            config_kwargs["plant_slow_only"] = True
            config_kwargs["plant_gait_support"] = True
            config_kwargs["source_slow_xy_m_s"] = 0.10
            config_kwargs["object_hull_clearance_weight"] = 0.0
            config_kwargs["leg_volume_clearance_weight"] = 400.0
            config_kwargs["leg_volume_clearance_margin_m"] = 0.0
            config_kwargs["leg_hull_weight"] = 1200.0
            config_kwargs["torso_volume_clearance_weight"] = 200.0
            config_kwargs["torso_volume_clearance_margin_m"] = 0.0
            config_kwargs["object_clearance_margin_m"] = 0.04
            config_kwargs["object_clearance_weight"] = 200.0
            config_kwargs["foot_slide_weight"] = 1200.0
            config_kwargs["foot_slide_hard"] = True
            config_kwargs["swing_lift_weight"] = 80.0
            config_kwargs["swing_height_m"] = 0.045
            config_kwargs["swing_knee_weight"] = 0.0
            config_kwargs["gait_knee_fidelity_weight"] = 90.0
            config_kwargs["gait_knee_bias_rad"] = 0.08
            config_kwargs["gait_knee_min_rad"] = 0.30
            config_kwargs["gait_knee_max_rad"] = 1.10
            config_kwargs["grasp_up_frac"] = 0.4
            config_kwargs["plant_exclusive"] = False
            config_kwargs["hard_plant_weight"] = 800.0
            config_kwargs["squat_knee_weight"] = 0.0
            config_kwargs["torso_upright_weight"] = 18.0
            config_kwargs["pose_fidelity_weight"] = 16.0
            config_kwargs["palmar_contact"] = True
            config_kwargs["palm_facing_weight"] = 25.0
            config_kwargs["finger_up_weight"] = 20.0
            config_kwargs["palm_offset_m"] = 0.050
            config_kwargs["plant_stop_grad"] = False
            config_kwargs["stance_lock_weight"] = 1200.0
            config_kwargs["putdown_dual_plant"] = True
            config_kwargs["putdown_phase_min"] = 4
            config_kwargs["putdown_phase_max"] = 4
            config_kwargs["putdown_tail_frames"] = 18
            config_kwargs["squat_stance_width_m"] = 0.24
            config_kwargs["putdown_root_lock_weight"] = 0.0
            config_kwargs["knee_symmetry_weight"] = 30.0
            config_kwargs["squat_knee_min_weight"] = 60.0
            config_kwargs["squat_knee_min_rad"] = 0.8
            config_kwargs["both_soles_weight"] = 200.0
            config_kwargs["stance_width_weight"] = 20.0
            config_kwargs["planted_root_slip_weight"] = 400.0
            config_kwargs["object_hull_clearance_weight"] = 600.0
            config_kwargs["hull_slack_m"] = 0.0
            config_kwargs["hull_clearance_margin_m"] = 0.006
            config_kwargs["hull_use_contact_mask"] = False
            config_kwargs["hull_mask_aabb"] = False
            config_kwargs["hull_include_feet"] = False
            config_kwargs["hand_volume_clearance_weight"] = 1200.0
            config_kwargs["hand_volume_clearance_margin_m"] = 0.008
            config_kwargs["finger_up_weight"] = 60.0
            config_kwargs["palm_facing_weight"] = 50.0
            config_kwargs["root_fidelity_weight"] = 35.0
            config_kwargs["pose_fidelity_weight"] = 25.0
            config_kwargs["jerk_weight"] = 40.0
            config_kwargs["learning_rate"] = 0.010
            config_kwargs["leg_laterality_weight"] = 250.0
            config_kwargs["leg_laterality_ankle_m"] = 0.02
            config_kwargs["leg_laterality_knee_m"] = 0.04
            config_kwargs["self_leg_clearance_weight"] = 400.0
            config_kwargs["self_leg_clearance_m"] = 0.06
            config_kwargs["hip_yaw_limit_weight"] = 150.0
            config_kwargs["hip_yaw_max_rad"] = 0.80
            config_kwargs["hip_roll_limit_weight"] = 80.0
            config_kwargs["hip_roll_max_rad"] = 0.45
            config_kwargs["swing_outward_weight"] = 250.0
            config_kwargs["swing_outward_slack_m"] = 0.04
            config_kwargs["root_speed_cap_weight"] = 0.0
            config_kwargs["foot_speed_cap_weight"] = 0.0
            config_kwargs["approach_ground_weight"] = 100.0
            config_kwargs["source_approach_root_weight"] = 240.0
            config_kwargs["swing_lift_min_root_m_s"] = 0.15
            config_kwargs["hull_clearance_margin_m"] = 0.008
            config_kwargs["palm_offset_m"] = 0.052
            config_kwargs["arm_hull_weight"] = 120.0
            config_kwargs["object_hull_clearance_weight"] = 800.0
            config_kwargs["leg_hull_weight"] = 1200.0
            if output_root in (
                Path("outputs/coupled"),
                Path("outputs/balanced"),
                Path("outputs/balanced_clearance_v13"),
                Path("outputs/balanced_clearance_v20"),
                Path("outputs/balanced_clearance_v21"),
                Path("outputs/balanced_clearance_v22"),
                Path("outputs/balanced_clearance_v23"),
                Path("outputs/balanced_clearance_v24"),
                Path("outputs/balanced_clearance_v25"),
                Path("outputs/balanced_clearance_v26"),
                Path("outputs/balanced_clearance_v27"),
                Path("outputs/balanced_clearance_v28"),
                Path("outputs/balanced_clearance_v29"),
                Path("outputs/balanced_clearance_v30"),
                Path("outputs/balanced_clearance_v31"),
                Path("outputs/balanced_clearance_v32"),
                Path("outputs/balanced_clearance_v33"),
                Path("outputs/balanced_clearance_v34"),
                Path("outputs/balanced_clearance_v35"),
                Path("outputs/balanced_clearance_v36"),
                Path("outputs/balanced_clearance_v37"),
                Path("outputs/balanced_clearance_v38"),
                Path("outputs/balanced_clearance_v39"),
                Path("outputs/balanced_clearance_v40"),
            ):
                output_root = Path("outputs/balanced_clearance_v41")
    if from_v13 and baseline_root == Path("outputs/independent"):
        baseline_root = Path("outputs/balanced_clearance_v13_smooth")
    if from_v20 and baseline_root == Path("outputs/independent"):
        baseline_root = Path("outputs/balanced_clearance_v20")
    if from_v23 and baseline_root == Path("outputs/independent"):
        baseline_root = Path("outputs/balanced_clearance_v23_smooth")
    if from_v24 and baseline_root == Path("outputs/independent"):
        baseline_root = Path("outputs/balanced_clearance_v24_smooth")
    if from_v30 and baseline_root == Path("outputs/independent"):
        baseline_root = Path("outputs/balanced_clearance_v30_smooth")
    if from_v31 and baseline_root == Path("outputs/independent"):
        baseline_root = Path("outputs/balanced_clearance_v31_smooth")
    if from_v32 and baseline_root == Path("outputs/independent"):
        baseline_root = Path("outputs/balanced_clearance_v32_smooth")
    if from_v33 and baseline_root == Path("outputs/independent"):
        baseline_root = Path("outputs/balanced_clearance_v33_smooth")
    if from_v34 and baseline_root == Path("outputs/independent"):
        baseline_root = Path("outputs/balanced_clearance_v34_smooth")
    if from_v35 and baseline_root == Path("outputs/independent"):
        baseline_root = Path("outputs/balanced_clearance_v35_smooth")
    if from_v36 and baseline_root == Path("outputs/independent"):
        baseline_root = Path("outputs/balanced_clearance_v36_smooth")
    if from_v39 and baseline_root == Path("outputs/independent"):
        baseline_root = Path("outputs/balanced_clearance_v39_smooth")
    if from_v40 and baseline_root == Path("outputs/independent"):
        baseline_root = Path("outputs/balanced_clearance_v40_smooth")
    outputs, history = retarget_coupled(
        motion_root,
        sequence_id,
        object_models_root,
        baseline_root,
        urdf_path,
        contact_analysis_root / f"{stem}.npz",
        CoupledRetargetConfig(**config_kwargs),
    )
    output_dir = output_root / stem
    save_coupled_results(outputs, history, output_dir)
    console.print(f"initial_loss={history[0]:.6f}")
    console.print(f"final_loss={history[-1]:.6f}")
    console.print(f"Wrote {output_dir}")


@app.command("retarget-paper-join")
def retarget_paper_join_command(
    sequence_id: str = typer.Argument(help="CORE4D join sequence ID."),
    variant: str = typer.Option(..., "--variant", help="Controlled variant J0--J4."),
    motion_root: Path = typer.Option(Path("data/core4d/motions")),
    object_models_root: Path = typer.Option(Path("data/core4d/object_models")),
    baseline_root: Path = typer.Option(Path("outputs/independent")),
    contact_analysis_root: Path = typer.Option(Path("outputs/contact_analysis")),
    urdf_path: Path = typer.Option(
        Path("third_party/unitree_ros/robots/g1_description/g1_29dof_rev_1_0.urdf"),
        exists=True,
        dir_okay=False,
    ),
    output_root: Path = typer.Option(Path("outputs/paper_eval_join/ablation_018")),
    iterations: int = typer.Option(600, min=1),
    override: list[str] = typer.Option(
        [],
        "--override",
        help=(
            "Optional KEY=VALUE applied on top of the preset (J1--J4 only). "
            "Used for single-term ON/OFF controls; recorded in config.json."
        ),
    ),
    tag: str | None = typer.Option(
        None,
        "--tag",
        help="Optional output folder name instead of the variant key.",
    ),
    solve_robot: int | None = typer.Option(
        None,
        "--solve-robot",
        min=1,
        max=2,
        help=(
            "OA-Independent mode: optimize only this robot with the identical "
            "per-robot objective (writes robotN.npz only; run both robots to "
            "complete the result)."
        ),
    ),
    term_log: bool = typer.Option(
        False,
        "--term-log",
        help="Write per-iteration weighted term values and cross-agent gradient norms.",
    ),
    root_offset: list[str] = typer.Option(
        [],
        "--root-offset",
        help="Stress test: ROBOT:DX:DY metres added to the initial root XY (pre-registered).",
    ),
) -> None:
    """Run one controlled, non-destructive Join paper variant."""
    import csv
    import dataclasses
    import shutil
    import sys

    from .retarget.coupled import retarget_coupled, save_coupled_results
    from .retarget.paper_presets import (
        PAPER_VARIANTS,
        paper_join_config,
        paper_join_config_dict,
    )

    key = variant.upper()
    if key not in PAPER_VARIANTS:
        raise typer.BadParameter(f"variant must be one of {', '.join(PAPER_VARIANTS)}")
    if override and key == "J0":
        raise typer.BadParameter("--override is only valid for coupled variants J1--J4")
    overrides: dict[str, object] = {}
    for item in override:
        if "=" not in item:
            raise typer.BadParameter(f"--override expects KEY=VALUE, got {item!r}")
        name, raw_value = item.split("=", 1)
        try:
            overrides[name.strip()] = json.loads(raw_value)
        except json.JSONDecodeError:
            # Bare string value (e.g. a path whose quotes were stripped by xargs).
            overrides[name.strip()] = raw_value.strip()
    offsets: dict[int, tuple[float, float]] = {}
    for item in root_offset:
        parts = item.split(":")
        if len(parts) != 3:
            raise typer.BadParameter(f"--root-offset expects ROBOT:DX:DY, got {item!r}")
        offsets[int(parts[0])] = (float(parts[1]), float(parts[2]))
    stem = sequence_id.replace("/", "_")
    variant_root = output_root / (tag or key)
    raw_root = variant_root / "raw"
    output_dir = raw_root / stem
    if solve_robot is None and output_dir.exists():
        raise typer.BadParameter(
            f"Refusing to overwrite existing paper result: {output_dir}"
        )
    if solve_robot is not None and (output_dir / f"robot{solve_robot}.npz").exists():
        raise typer.BadParameter(
            f"Refusing to overwrite existing paper result: {output_dir / f'robot{solve_robot}.npz'}"
        )
    if solve_robot is not None and key == "J0":
        raise typer.BadParameter("--solve-robot is only valid for coupled variants J1--J4")
    if key == "J0":
        output_dir.mkdir(parents=True, exist_ok=False)
        baseline_dir = baseline_root / stem
        for person in (1, 2):
            source = baseline_dir / f"robot{person}.npz"
            if not source.exists():
                raise FileNotFoundError(source)
            shutil.copy2(source, output_dir / source.name)
        history = None
    else:
        config = paper_join_config(key, iterations=iterations)
        if overrides:
            valid = {field.name for field in dataclasses.fields(config)}
            unknown = sorted(set(overrides) - valid)
            if unknown:
                raise typer.BadParameter(f"Unknown config fields: {unknown}")
            config = dataclasses.replace(config, **overrides)
        if solve_robot is not None:
            config = dataclasses.replace(config, solve_robots=(solve_robot,))
        log_records: list[dict[str, float]] | None = [] if term_log else None
        outputs, history = retarget_coupled(
            motion_root,
            sequence_id,
            object_models_root,
            baseline_root,
            urdf_path,
            contact_analysis_root / f"{stem}.npz",
            config,
            initial_root_offsets_xy=offsets or None,
            term_log=log_records,
        )
        save_coupled_results(outputs, history, output_dir)
        if log_records:
            suffix = f"_robot{solve_robot}" if solve_robot is not None else ""
            log_path = variant_root / f"term_log_{stem}{suffix}.csv"
            fieldnames = sorted({key for record in log_records for key in record})
            with log_path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(log_records)
    variant_root.mkdir(parents=True, exist_ok=True)
    provenance = paper_join_config_dict(key, iterations=iterations)
    if overrides or solve_robot is not None:
        provenance["config"] = dataclasses.asdict(config)
        provenance["overrides"] = overrides
        provenance["base_variant"] = key
    if solve_robot is not None:
        provenance["solve_robot"] = solve_robot
    if offsets:
        provenance["initial_root_offsets_xy"] = {str(k): list(v) for k, v in offsets.items()}
    provenance.update(
        {
            "sequence_id": sequence_id,
            "baseline_root": str(baseline_root),
            "output_dir": str(output_dir),
            "command": " ".join(sys.argv),
        }
    )
    config_name = "config.json" if solve_robot is None else f"config_robot{solve_robot}.json"
    (variant_root / config_name).write_text(
        json.dumps(provenance, indent=2) + "\n", encoding="utf-8"
    )
    if history is not None:
        console.print(f"initial_loss={history[0]:.6f}")
        console.print(f"final_loss={history[-1]:.6f}")
    console.print(f"Wrote {output_dir}")


@app.command("smooth-trajectories")
def smooth_trajectories_command(
    sequence_id: str = typer.Argument(help="CORE4D sequence ID."),
    result_root: Path = typer.Option(Path("outputs/balanced")),
    output_root: Path = typer.Option(Path("outputs/balanced_smooth")),
    urdf_path: Path = typer.Option(
        Path("third_party/unitree_ros/robots/g1_description/g1_29dof_rev_1_0.urdf"),
        exists=True,
        dir_okay=False,
    ),
    sigma: float = typer.Option(1.0, min=0.0),
    arm_sigma: float | None = typer.Option(
        None,
        help="Optional smaller Gaussian sigma for shoulder/elbow/wrist joints.",
    ),
    root_sigma: float | None = typer.Option(
        None,
        help="Optional smaller Gaussian sigma for root translation and yaw.",
    ),
    clearance_recipe: bool = typer.Option(
        False,
        "--clearance-recipe",
        help="After a --clearance retarget: arm_sigma=0.35 and root_sigma=0 unless set.",
    ),
    lift_soles: bool | None = typer.Option(
        None,
        "--lift-soles/--no-lift-soles",
        help="Raise root z after smooth so sole samples are not underground. Default on with --clearance-recipe.",
    ),
    ankle_target: float | None = typer.Option(
        None,
        help="Optional per-frame root-z nudge so mean ankle keypoints match this height.",
    ),
) -> None:
    """Gaussian-smooth joints/root and refresh forward-kinematics keypoints."""
    from .retarget.smooth import resolve_smooth_sigmas, smooth_robot_npz

    arm_sigma, root_sigma = resolve_smooth_sigmas(
        arm_sigma=arm_sigma,
        root_sigma=root_sigma,
        clearance_recipe=clearance_recipe,
    )
    if lift_soles is None:
        lift_soles = clearance_recipe
    stem = sequence_id.replace("/", "_")
    source = result_root / stem
    output = output_root / stem
    for person in (1, 2):
        smooth_robot_npz(
            source / f"robot{person}.npz",
            urdf_path,
            output / f"robot{person}.npz",
            sigma=sigma,
            arm_sigma=arm_sigma,
            root_sigma=root_sigma,
            ankle_target_z=ankle_target,
            lift_soles=lift_soles,
        )
    extra = ""
    if arm_sigma is not None:
        extra += f" arm_sigma={arm_sigma:g}"
    if root_sigma is not None:
        extra += f" root_sigma={root_sigma:g}"
    if lift_soles:
        extra += " lift_soles"
    console.print(f"Wrote {output} sigma={sigma:g}{extra}")


@app.command("aggregate")
def aggregate_command(
    independent_dir: Path = typer.Option(Path("outputs/metrics/independent")),
    coupled_dir: Path = typer.Option(Path("outputs/metrics/coupled")),
    output_path: Path = typer.Option(Path("outputs/metrics/summary.json")),
    baseline_name: str = typer.Option("independent"),
    method_name: str = typer.Option("coupled"),
) -> None:
    """Aggregate paired baseline and method results across sequences."""
    from .aggregate import aggregate_comparison, write_aggregate

    metrics = aggregate_comparison(
        independent_dir,
        coupled_dir,
        baseline_name=baseline_name,
        method_name=method_name,
    )
    write_aggregate(metrics, output_path)
    console.print(json.dumps(metrics, indent=2))
    console.print(f"Wrote {output_path}")


@app.command("plot-comparison")
def plot_comparison_command(
    summary_path: Path = typer.Option(
        Path("outputs/metrics/summary.json"),
        exists=True,
        dir_okay=False,
    ),
    output_path: Path = typer.Option(Path("outputs/metrics/comparison.png")),
) -> None:
    """Plot aggregate independent-versus-coupled metrics."""
    plot_metric_comparison(summary_path, output_path)
    console.print(f"Wrote {output_path}")


@app.command("render-g1-meshes")
def render_g1_meshes_command(
    sequence_id: str = typer.Argument(help="CORE4D sequence ID."),
    motion_root: Path = typer.Option(Path("data/core4d/motions")),
    object_models_root: Path = typer.Option(Path("data/core4d/object_models")),
    result_root: Path = typer.Option(Path("outputs/coupled")),
    g1_xml: Path = typer.Option(
        Path("third_party/unitree_ros/robots/g1_description/g1_29dof_rev_1_0.xml"),
        exists=True,
        dir_okay=False,
    ),
    output_path: Path = typer.Option(Path("videos/04_dual_g1_meshes.mp4")),
    frame_step: int = typer.Option(1, min=1),
) -> None:
    """Render actual official G1 meshes with the retargeted shared object."""
    from .contact import resolve_object_model
    from .mujoco_scene import build_dual_g1_scene_xml, render_dual_g1_video

    sequence_dir = motion_root.joinpath(*sequence_id.split("/"))
    object_pose_path = sequence_dir / "smooth_objposes_so3.npy"
    if not object_pose_path.exists():
        object_pose_path = sequence_dir / "smooth_objposes.npy"
    object_model = resolve_object_model(sequence_dir, object_models_root)
    stem = sequence_id.replace("/", "_")
    scene_xml = Path("outputs/mujoco") / stem / "dual_g1_render.xml"
    build_dual_g1_scene_xml(g1_xml, object_model, scene_xml)
    result_dir = result_root / stem
    render_dual_g1_video(
        scene_xml,
        np.load(object_pose_path),
        result_dir / "robot1.npz",
        result_dir / "robot2.npz",
        output_path,
        frame_step=frame_step,
    )
    console.print(f"Wrote {output_path}")


@app.command("simulate-pd")
def simulate_pd_command(
    sequence_id: str = typer.Argument(help="CORE4D sequence ID."),
    motion_root: Path = typer.Option(Path("data/core4d/motions")),
    object_models_root: Path = typer.Option(Path("data/core4d/object_models")),
    result_root: Path = typer.Option(Path("outputs/coupled")),
    g1_xml: Path = typer.Option(
        Path("third_party/unitree_ros/robots/g1_description/g1_29dof_rev_1_0.xml"),
        exists=True,
        dir_okay=False,
    ),
    output_root: Path = typer.Option(Path("outputs/pd_tracking")),
    output_name: str | None = typer.Option(
        None,
        "--output-name",
        help="Short exam folder name. Use when the auto suffix exceeds NAME_MAX.",
    ),
    hold_first_frame_seconds: float = typer.Option(0.0, min=0.0),
    pin_base: bool = typer.Option(False, "--pin-base"),
    gravity_compensation: bool = typer.Option(False, "--gravity-comp"),
    com_feedback: bool = typer.Option(False, "--com-feedback"),
    time_scale: float = typer.Option(1.0, min=0.1),
    com_kp: float = typer.Option(2500.0, min=0.0),
    com_mode: str = typer.Option("jacobian"),
    weld_feet: bool = typer.Option(False, "--weld-feet"),
    foot_boxes: bool = typer.Option(False, "--foot-boxes"),
    contact_compensation: bool = typer.Option(False, "--contact-comp"),
    sim_contacts: bool = typer.Option(False, "--sim-contacts"),
    position_ctrl: bool = typer.Option(False, "--position-ctrl"),
    no_project: bool = typer.Option(False, "--no-project"),
    online_com: bool = typer.Option(True, "--online-com/--no-online-com"),
    freeze_legs: bool = typer.Option(False, "--freeze-legs"),
    freeze_waist: bool = typer.Option(False, "--freeze-waist"),
    freeze_ankles: bool = typer.Option(False, "--freeze-ankles"),
    freeze_stance_foot: bool = typer.Option(False, "--freeze-stance-foot"),
    freeze_stance_latch: bool = typer.Option(
        True, "--freeze-stance-latch/--no-freeze-stance-latch"
    ),
    freeze_swing_sagittal: bool = typer.Option(False, "--freeze-swing-sagittal"),
    hold_swing_sole: bool = typer.Option(False, "--hold-swing-sole"),
    hold_swing_com: bool = typer.Option(False, "--hold-swing-com"),
    stance_foot_margin: float = typer.Option(0.008, "--stance-foot-margin", min=0.0),
    project_target: bool = typer.Option(False, "--project-target"),
    settle_seconds: float = typer.Option(0.0, "--settle-seconds", min=0.0),
    com_integral: bool = typer.Option(False, "--com-integral"),
    com_ki: float = typer.Option(2.0, "--com-ki", min=0.0),
    residual: Path | None = typer.Option(None, "--residual"),
    residual_joints: str = typer.Option("ankle", "--residual-joints"),
    hip_pitch_bias: float = typer.Option(0.0, "--hip-pitch-bias"),
    noslip_iterations: int | None = typer.Option(None, "--noslip-iterations"),
    contact_friction: float | None = typer.Option(None, "--contact-friction"),
    stagger_split: float | None = typer.Option(None, "--stagger-split"),
    cone: str | None = typer.Option(None, "--cone"),
    slip_clutch: bool = typer.Option(False, "--slip-clutch"),
    slip_speed: float = typer.Option(0.05, "--slip-speed", min=0.0),
    slip_delay: float = typer.Option(0.4, "--slip-delay", min=0.0),
    slip_robots: str = typer.Option("both", "--slip-robots"),
    slip_gain: float = typer.Option(0.0, "--slip-gain"),
    solver_iterations: int | None = typer.Option(None, "--solver-iterations"),
    impratio: float | None = typer.Option(None, "--impratio"),
    sole_condim: int | None = typer.Option(None, "--sole-condim"),
    r1_sole_condim: int | None = typer.Option(None, "--r1-sole-condim"),
    r1_sticky_floor: bool = typer.Option(False, "--r1-sticky-floor"),
    r2_sticky_floor: bool = typer.Option(False, "--r2-sticky-floor"),
    r2_sticky_condim: int = typer.Option(6, "--r2-sticky-condim"),
    slip_kp_scale: float = typer.Option(1.0, "--slip-kp-scale"),
    waist_kp: float | None = typer.Option(None, "--waist-kp"),
    r2_hip_roll_bias: float = typer.Option(0.0, "--r2-hip-roll-bias"),
    r2_com_roll_gain: float = typer.Option(0.0, "--r2-com-roll-gain"),
    r2_com_roll_kd: float = typer.Option(0.0, "--r2-com-roll-kd"),
    r2_com_ankle_gain: float = typer.Option(0.0, "--r2-com-ankle-gain"),
    r2_com_ki: float = typer.Option(0.0, "--r2-com-ki"),
    r2_com_pitch_gain: float = typer.Option(0.0, "--r2-com-pitch-gain"),
    r2_com_y_ref: float = typer.Option(0.0, "--r2-com-y-ref"),
    r2_com_x_ref: float = typer.Option(0.0, "--r2-com-x-ref"),
    r2_late_y_ref: float | None = typer.Option(None, "--r2-late-y-ref"),
    r2_com_support: str = typer.Option("midpoint", "--r2-com-support"),
    r2_com_ref_latch: bool = typer.Option(False, "--r2-com-ref-latch"),
    r2_com_ref_delay: float = typer.Option(0.0, "--r2-com-ref-delay", min=0.0),
    r2_arm_kp: float | None = typer.Option(None, "--r2-arm-kp"),
    r2_waist_roll_bias: float = typer.Option(0.0, "--r2-waist-roll-bias"),
    r2_waist_latch: bool = typer.Option(False, "--r2-waist-latch"),
    r2_waist_latch_delay: float = typer.Option(1.0, "--r2-waist-latch-delay", min=0.0),
    r2_waist_roll_kp: float | None = typer.Option(None, "--r2-waist-roll-kp"),
    r2_level_soles: bool = typer.Option(False, "--r2-level-soles"),
    r2_project_com: bool = typer.Option(False, "--r2-project-com"),
    ankle_kp: float | None = typer.Option(None, "--ankle-kp"),
    r2_ankle_kp: float | None = typer.Option(None, "--r2-ankle-kp"),
    r2_leg_kp: float | None = typer.Option(None, "--r2-leg-kp"),
    r2_ankle_pitch_bias: float = typer.Option(0.0, "--r2-ankle-pitch-bias"),
    wrist_kp_override: float | None = typer.Option(None, "--wrist-kp"),
    r2_pose_latch: bool = typer.Option(False, "--r2-pose-latch"),
    r2_pose_latch_delay: float = typer.Option(2.0, "--r2-pose-latch-delay", min=0.0),
    r2_pose_latch_legs: bool = typer.Option(False, "--r2-pose-latch-legs"),
    r2_target_follow: float = typer.Option(0.0, "--r2-target-follow"),
    r2_plant_foot: bool = typer.Option(False, "--r2-plant-foot"),
    r2_late_delay: float = typer.Option(4.0, "--r2-late-delay", min=0.0),
    r2_late_ankle_pitch: float = typer.Option(0.0, "--r2-late-ankle-pitch"),
    r2_late_ankle_delay: float | None = typer.Option(None, "--r2-late-ankle-delay"),
    r2_late_ankle_both: bool = typer.Option(False, "--r2-late-ankle-both"),
    r2_late_roll_gain: float | None = typer.Option(None, "--r2-late-roll-gain"),
    r2_late_pitch_gain: float | None = typer.Option(None, "--r2-late-pitch-gain"),
    r2_late_pitch_delay: float | None = typer.Option(None, "--r2-late-pitch-delay"),
    r2_late_hip_pitch: float = typer.Option(0.0, "--r2-late-hip-pitch"),
    r2_late2_hip_pitch: float = typer.Option(0.0, "--r2-late2-hip-pitch"),
    r2_late2_delay: float | None = typer.Option(None, "--r2-late2-delay"),
    r2_late3_hip_pitch: float = typer.Option(0.0, "--r2-late3-hip-pitch"),
    r2_late3_delay: float | None = typer.Option(None, "--r2-late3-delay"),
    r2_late_knee: float = typer.Option(0.0, "--r2-late-knee"),
    r1_late_delay: float = typer.Option(4.0, "--r1-late-delay", min=0.0),
    r1_late_hip_pitch: float = typer.Option(0.0, "--r1-late-hip-pitch"),
    r1_late2_hip_pitch: float = typer.Option(0.0, "--r1-late2-hip-pitch"),
    r1_late2_delay: float | None = typer.Option(None, "--r1-late2-delay"),
    r1_late3_hip_pitch: float = typer.Option(0.0, "--r1-late3-hip-pitch"),
    r1_late3_delay: float | None = typer.Option(None, "--r1-late3-delay"),
    r1_late_pitch_gain: float | None = typer.Option(None, "--r1-late-pitch-gain"),
    r1_late_pitch_delay: float | None = typer.Option(None, "--r1-late-pitch-delay"),
    arm_ease_seconds: float = typer.Option(0.0, "--arm-ease-seconds", min=0.0),
    arm_scale: float = typer.Option(1.0, "--arm-scale"),
    leg_ease_seconds: float = typer.Option(0.0, "--leg-ease-seconds", min=0.0),
    leg_scale: float = typer.Option(1.0, "--leg-scale"),
    freeze_object: bool = typer.Option(False, "--freeze-object"),
    r2_arm_com_comp: float = typer.Option(0.0, "--r2-arm-com-comp"),
    com_roll_both: bool = typer.Option(False, "--com-roll-both"),
    timestep: float = typer.Option(0.002, "--timestep", min=0.0002),
    leg_kp: float = typer.Option(80.0, min=1.0),
    arm_kp: float = typer.Option(30.0, min=1.0),
) -> None:
    """Run an open-loop joint-PD feasibility check with free or pinned bases."""
    from .contact import resolve_object_model
    from .mujoco_scene import build_dual_g1_scene_xml
    from .pd_tracking import PDTrackingConfig, save_pd_tracking, simulate_pd_tracking
    from .residual import load_residual_gains

    sequence_dir = motion_root.joinpath(*sequence_id.split("/"))
    object_pose_path = sequence_dir / "smooth_objposes_so3.npy"
    if not object_pose_path.exists():
        object_pose_path = sequence_dir / "smooth_objposes.npy"
    object_model = resolve_object_model(sequence_dir, object_models_root)
    stem = sequence_id.replace("/", "_")
    if position_ctrl:
        foot_boxes = True
        if leg_kp == 80.0:
            leg_kp = 200.0
        if arm_kp == 30.0:
            arm_kp = 40.0
        com_mode = "heuristic"
        if com_kp == 2500.0:
            com_kp = 2.5
    if weld_feet:
        scene_name = "dual_g1_pd_weld_feet.xml"
    elif position_ctrl:
        if r1_sticky_floor or r2_sticky_floor:
            scene_name = "dual_g1_pd_stance_sticky.xml"
        else:
            scene_name = "dual_g1_pd_stance.xml"
    elif foot_boxes:
        scene_name = "dual_g1_pd_feet.xml"
    else:
        scene_name = "dual_g1_pd.xml"
    scene_xml = Path("outputs/mujoco") / stem / scene_name
    build_dual_g1_scene_xml(
        g1_xml,
        object_model,
        scene_xml,
        weld_feet=weld_feet,
        foot_boxes=foot_boxes,
        position_actuators=position_ctrl,
        stiff_contacts=position_ctrl,
        leg_kp=leg_kp,
        arm_kp=arm_kp,
        r1_sticky_floor=r1_sticky_floor,
        r2_sticky_floor=r2_sticky_floor,
        r2_sticky_condim=r2_sticky_condim,
    )
    result_dir = result_root / stem
    if sim_contacts:
        contact_compensation = True
    contact_source = "sim" if sim_contacts else "virtual"
    arrays, metrics = simulate_pd_tracking(
        scene_xml,
        np.load(object_pose_path),
        result_dir / "robot1.npz",
        result_dir / "robot2.npz",
        PDTrackingConfig(
            hold_first_frame_seconds=hold_first_frame_seconds,
            pin_base=pin_base,
            gravity_compensation=gravity_compensation,
            com_feedback=com_feedback,
            time_scale=time_scale,
            com_kp=com_kp,
            com_kd=0.25 if position_ctrl else 200.0,
            com_mode=com_mode,
            weld_feet=weld_feet,
            contact_compensation=contact_compensation and not position_ctrl,
            contact_source=contact_source,
            control_mode="position" if position_ctrl else "torque",
            lift_feet=position_ctrl,
            project_com=position_ctrl and not no_project,
            online_com=position_ctrl and online_com,
            freeze_legs=freeze_legs,
            freeze_waist=freeze_waist,
            freeze_ankles=freeze_ankles,
            freeze_stance_foot=freeze_stance_foot,
            freeze_stance_latch=freeze_stance_latch,
            freeze_swing_sagittal=freeze_swing_sagittal,
            hold_swing_sole=hold_swing_sole,
            hold_swing_com=hold_swing_com,
            stance_foot_margin=stance_foot_margin,
            project_target=project_target,
            settle_seconds=settle_seconds,
            com_integral=com_integral,
            com_ki=com_ki,
            residual_gains=load_residual_gains(residual) if residual is not None else None,
            residual_joints=residual_joints,
            hip_pitch_bias=hip_pitch_bias,
            noslip_iterations=noslip_iterations,
            contact_friction=contact_friction,
            stagger_split=stagger_split,
            cone=cone,
            slip_clutch=slip_clutch,
            slip_speed=slip_speed,
            slip_delay=slip_delay,
            slip_robots=slip_robots,
            slip_gain=slip_gain,
            solver_iterations=solver_iterations,
            impratio=impratio,
            sole_condim=sole_condim,
            r1_sole_condim=r1_sole_condim,
            slip_kp_scale=slip_kp_scale,
            r2_hip_roll_bias=r2_hip_roll_bias,
            r2_com_roll_gain=r2_com_roll_gain,
            r2_com_roll_kd=r2_com_roll_kd,
            r2_com_ankle_gain=r2_com_ankle_gain,
            r2_com_ki=r2_com_ki,
            r2_com_pitch_gain=r2_com_pitch_gain,
            r2_com_y_ref=r2_com_y_ref,
            r2_com_x_ref=r2_com_x_ref,
            r2_late_y_ref=r2_late_y_ref,
            r2_com_support=r2_com_support,
            r2_com_ref_latch=r2_com_ref_latch,
            r2_com_ref_delay=r2_com_ref_delay,
            r2_arm_kp=r2_arm_kp,
            r2_waist_roll_bias=r2_waist_roll_bias,
            r2_waist_latch=r2_waist_latch,
            r2_waist_latch_delay=r2_waist_latch_delay,
            r2_waist_roll_kp=r2_waist_roll_kp,
            r2_level_soles=r2_level_soles,
            r2_project_com=r2_project_com,
            ankle_kp=ankle_kp,
            r2_ankle_kp=r2_ankle_kp,
            r2_leg_kp=r2_leg_kp,
            r2_ankle_pitch_bias=r2_ankle_pitch_bias,
            wrist_kp_override=wrist_kp_override,
            r2_pose_latch=r2_pose_latch or r2_pose_latch_legs,
            r2_pose_latch_delay=r2_pose_latch_delay,
            r2_pose_latch_legs=r2_pose_latch_legs,
            r2_target_follow=r2_target_follow,
            r2_plant_foot=r2_plant_foot,
            r2_late_delay=r2_late_delay,
            r2_late_ankle_pitch=r2_late_ankle_pitch,
            r2_late_ankle_delay=r2_late_ankle_delay,
            r2_late_ankle_both=r2_late_ankle_both,
            r2_late_roll_gain=r2_late_roll_gain,
            r2_late_pitch_gain=r2_late_pitch_gain,
            r2_late_pitch_delay=r2_late_pitch_delay,
            r2_late_hip_pitch=r2_late_hip_pitch,
            r2_late2_hip_pitch=r2_late2_hip_pitch,
            r2_late2_delay=r2_late2_delay,
            r2_late3_hip_pitch=r2_late3_hip_pitch,
            r2_late3_delay=r2_late3_delay,
            r2_late_knee=r2_late_knee,
            r1_late_delay=r1_late_delay,
            r1_late_hip_pitch=r1_late_hip_pitch,
            r1_late2_hip_pitch=r1_late2_hip_pitch,
            r1_late2_delay=r1_late2_delay,
            r1_late3_hip_pitch=r1_late3_hip_pitch,
            r1_late3_delay=r1_late3_delay,
            r1_late_pitch_gain=r1_late_pitch_gain,
            r1_late_pitch_delay=r1_late_pitch_delay,
            arm_ease_seconds=arm_ease_seconds,
            arm_scale=arm_scale,
            leg_ease_seconds=leg_ease_seconds,
            leg_scale=leg_scale,
            freeze_object=freeze_object,
            r2_arm_com_comp=r2_arm_com_comp,
            com_roll_both=com_roll_both,
            timestep=timestep,
            leg_kp=leg_kp,
            arm_kp=arm_kp,
            waist_kp=(
                waist_kp
                if waist_kp is not None
                else (80.0 if position_ctrl else 60.0)
            ),
        ),
    )
    suffix = ""
    if pin_base:
        suffix += "_pinned"
    if weld_feet:
        suffix += "_weldfeet"
    if position_ctrl:
        suffix += "_stance"
    elif foot_boxes:
        suffix += "_feet"
    if position_ctrl and online_com:
        suffix += "_online"
    if freeze_legs:
        suffix += "_freezelegs"
    if freeze_waist:
        suffix += "_freezewaist"
    if freeze_ankles:
        suffix += "_freezeankles"
    if freeze_stance_foot:
        suffix += "_freezestance"
        if not freeze_stance_latch:
            suffix += "dyn"
        if abs(stance_foot_margin - 0.008) > 1e-12:
            suffix += f"m{stance_foot_margin:g}"
    if freeze_swing_sagittal:
        suffix += "_swingsag"
    if hold_swing_sole:
        suffix += "_holdsole"
    if hold_swing_com:
        suffix += "_holdcom"
    if project_target:
        suffix += "_tgtproj"
    if settle_seconds > 0:
        suffix += f"_settle{settle_seconds:g}s"
    if com_integral:
        suffix += "_comint"
    if residual is not None:
        suffix += "_residual"
        if residual_joints != "ankle":
            suffix += f"_{residual_joints}"
    if abs(hip_pitch_bias) > 1e-12:
        suffix += f"_hipbias{hip_pitch_bias:g}"
    if noslip_iterations is not None:
        suffix += f"_noslip{noslip_iterations:g}"
    if contact_friction is not None:
        suffix += f"_mu{contact_friction:g}"
    if stagger_split is not None:
        suffix += f"_stagger{stagger_split:g}"
    if cone:
        suffix += f"_{cone}"
    if slip_clutch:
        suffix += "_clutch"
        if abs(slip_delay - 0.4) > 1e-12:
            suffix += f"{slip_delay:g}s"
        if slip_robots != "both":
            suffix += f"r{slip_robots}"
    if abs(slip_gain) > 1e-12:
        suffix += f"_slipg{slip_gain:g}"
    if solver_iterations is not None:
        suffix += f"_iter{solver_iterations:g}"
    if impratio is not None:
        suffix += f"_imp{impratio:g}"
    if sole_condim is not None:
        suffix += f"_condim{sole_condim:g}"
    if r1_sole_condim is not None:
        suffix += f"_r1condim{r1_sole_condim:g}"
    if r1_sticky_floor:
        suffix += "_r1sticky"
    if r2_sticky_floor:
        suffix += "_r2sticky"
        if r2_sticky_condim != 6:
            suffix += f"c{r2_sticky_condim:g}"
    if abs(slip_kp_scale - 1.0) > 1e-12:
        suffix += f"_kps{slip_kp_scale:g}"
    if waist_kp is not None:
        suffix += f"_waistkp{waist_kp:g}"
    if abs(r2_hip_roll_bias) > 1e-12:
        suffix += f"_r2roll{r2_hip_roll_bias:g}"
    if abs(r2_com_roll_gain) > 1e-12:
        suffix += f"_r2comroll{r2_com_roll_gain:g}"
    if abs(r2_com_roll_kd) > 1e-12:
        suffix += f"_r2comkd{r2_com_roll_kd:g}"
    if abs(r2_com_ankle_gain) > 1e-12:
        suffix += f"_r2ankleg{r2_com_ankle_gain:g}"
    if abs(r2_com_ki) > 1e-12:
        suffix += f"_r2ki{r2_com_ki:g}"
    if abs(r2_com_pitch_gain) > 1e-12:
        suffix += f"_r2pitchg{r2_com_pitch_gain:g}"
    if abs(r2_com_y_ref) > 1e-12:
        suffix += f"_r2yref{r2_com_y_ref:g}"
    if r2_late_y_ref is not None:
        suffix += f"_r2lateyref{r2_late_y_ref:g}"
    if abs(r2_com_x_ref) > 1e-12:
        suffix += f"_r2xref{r2_com_x_ref:g}"
    if r2_com_support != "midpoint":
        suffix += f"_r2sup{r2_com_support}"
    if r2_com_ref_latch:
        suffix += "_r2reflatch"
        if r2_com_ref_delay > 1e-12:
            suffix += f"{r2_com_ref_delay:g}s"
    if r2_arm_kp is not None:
        suffix += f"_r2armkp{r2_arm_kp:g}"
    if abs(r2_waist_roll_bias) > 1e-12:
        suffix += f"_r2wbias{r2_waist_roll_bias:g}"
    if r2_waist_latch:
        suffix += "_r2wlatch"
    if r2_waist_roll_kp is not None:
        suffix += f"_r2wkp{r2_waist_roll_kp:g}"
    if r2_level_soles:
        suffix += "_r2level"
    if r2_project_com:
        suffix += "_r2proj"
    if ankle_kp is not None:
        suffix += f"_anklekp{ankle_kp:g}"
    if r2_ankle_kp is not None:
        suffix += f"_r2anklekp{r2_ankle_kp:g}"
    if r2_leg_kp is not None:
        suffix += f"_r2legkp{r2_leg_kp:g}"
    if abs(r2_ankle_pitch_bias) > 1e-12:
        suffix += f"_r2apitch{r2_ankle_pitch_bias:g}"
    if wrist_kp_override is not None:
        suffix += f"_wristkp{wrist_kp_override:g}"
    if r2_pose_latch:
        suffix += "_r2platch"
        if abs(r2_pose_latch_delay - 2.0) > 1e-12:
            suffix += f"{r2_pose_latch_delay:g}s"
        if r2_pose_latch_legs:
            suffix += "legs"
    if abs(r2_target_follow) > 1e-12:
        suffix += f"_r2follow{r2_target_follow:g}"
    if r2_plant_foot:
        suffix += "_r2plant"
    if abs(r2_late_ankle_pitch) > 1e-12:
        suffix += f"_r2lateap{r2_late_ankle_pitch:g}"
        if r2_late_ankle_delay is not None:
            suffix += f"d{r2_late_ankle_delay:g}"
        if r2_late_ankle_both:
            suffix += "both"
    if r2_late_roll_gain is not None:
        suffix += f"_r2lateroll{r2_late_roll_gain:g}"
    if r2_late_pitch_gain is not None:
        suffix += f"_r2latepg{r2_late_pitch_gain:g}"
        if r2_late_pitch_delay is not None:
            suffix += f"d{r2_late_pitch_delay:g}"
    if abs(r2_late_hip_pitch) > 1e-12:
        suffix += f"_r2latehp{r2_late_hip_pitch:g}"
    if abs(r2_late2_hip_pitch) > 1e-12:
        suffix += f"_r2late2hp{r2_late2_hip_pitch:g}"
        if r2_late2_delay is not None:
            suffix += f"d{r2_late2_delay:g}"
    if abs(r2_late3_hip_pitch) > 1e-12:
        suffix += f"_r2late3hp{r2_late3_hip_pitch:g}"
        if r2_late3_delay is not None:
            suffix += f"d{r2_late3_delay:g}"
    if abs(r1_late_hip_pitch) > 1e-12:
        suffix += f"_r1latehp{r1_late_hip_pitch:g}"
    if abs(r1_late2_hip_pitch) > 1e-12:
        suffix += f"_r1late2hp{r1_late2_hip_pitch:g}"
        if r1_late2_delay is not None:
            suffix += f"d{r1_late2_delay:g}"
    if abs(r1_late3_hip_pitch) > 1e-12:
        suffix += f"_r1late3hp{r1_late3_hip_pitch:g}"
        if r1_late3_delay is not None:
            suffix += f"d{r1_late3_delay:g}"
    if r1_late_pitch_gain is not None:
        suffix += f"_r1latepg{r1_late_pitch_gain:g}"
        pg_delay = r1_late_delay if r1_late_pitch_delay is None else r1_late_pitch_delay
        if abs(pg_delay - 4.0) > 1e-12:
            suffix += f"d{pg_delay:g}"
    elif abs(r1_late_hip_pitch) > 1e-12 and abs(r1_late_delay - 4.0) > 1e-12:
        suffix += f"d{r1_late_delay:g}"
    if abs(r2_late_knee) > 1e-12:
        suffix += f"_r2latekn{r2_late_knee:g}"
    if arm_ease_seconds > 1e-12:
        suffix += f"_armease{arm_ease_seconds:g}s"
    if abs(arm_scale - 1.0) > 1e-12:
        suffix += f"_armsc{arm_scale:g}"
    if abs(leg_scale - 1.0) > 1e-12:
        suffix += f"_legsc{leg_scale:g}"
    if leg_ease_seconds > 1e-12:
        suffix += f"_legease{leg_ease_seconds:g}s"
    if freeze_object:
        suffix += "_obj0"
    if abs(r2_arm_com_comp) > 1e-12:
        suffix += f"_r2armcom{r2_arm_com_comp:g}"
    if com_roll_both:
        suffix += "_rollboth"
    if abs(timestep - 0.002) > 1e-12:
        suffix += f"_dt{timestep:g}"
    if contact_compensation and not position_ctrl:
        suffix += "_simcon" if sim_contacts else "_wbc"
    if gravity_compensation:
        suffix += "_grav"
    if com_feedback:
        suffix += "_com"
        if com_mode != "jacobian":
            suffix += f"_{com_mode}"
    if time_scale != 1.0:
        suffix += f"_x{time_scale:g}"
    if hold_first_frame_seconds > 0:
        suffix += f"_hold_{hold_first_frame_seconds:g}s"
    if leg_kp != 80.0 or arm_kp != 30.0:
        suffix += f"_legkp{leg_kp:g}"
    name = output_name or f"{stem}{suffix}"
    if len(name) > 240:
        import hashlib

        digest = hashlib.sha1(f"{stem}{suffix}".encode()).hexdigest()[:12]
        name = f"{stem}_{digest}"
    output_dir = output_root / name
    save_pd_tracking(
        arrays,
        metrics,
        output_dir / "trajectory.npz",
        output_dir / "metrics.json",
    )
    console.print(json.dumps(metrics, indent=2))
    console.print(f"Wrote {output_dir}")


@app.command("render-pd")
def render_pd_command(
    sequence_id: str = typer.Argument(help="CORE4D sequence ID."),
    motion_root: Path = typer.Option(Path("data/core4d/motions")),
    result_root: Path = typer.Option(Path("outputs/coupled")),
    pd_root: Path = typer.Option(Path("outputs/pd_tracking")),
    scene_root: Path = typer.Option(Path("outputs/mujoco")),
    output_path: Path = typer.Option(Path("videos/05_pd_tracking.mp4")),
) -> None:
    """Render the recorded free-base PD tracking result."""
    from .mujoco_scene import render_pd_tracking_video

    sequence_dir = motion_root.joinpath(*sequence_id.split("/"))
    object_pose_path = sequence_dir / "smooth_objposes_so3.npy"
    if not object_pose_path.exists():
        object_pose_path = sequence_dir / "smooth_objposes.npy"
    stem = sequence_id.replace("/", "_")
    result_dir = result_root / stem
    render_pd_tracking_video(
        scene_root / stem / "dual_g1_pd.xml",
        np.load(object_pose_path),
        pd_root / stem / "trajectory.npz",
        result_dir / "robot1.npz",
        result_dir / "robot2.npz",
        output_path,
    )
    console.print(f"Wrote {output_path}")


@app.command("inverse-dynamics")
def inverse_dynamics_command(
    sequence_id: str = typer.Argument(help="CORE4D sequence ID."),
    object_models_root: Path = typer.Option(Path("data/core4d/object_models")),
    motion_root: Path = typer.Option(Path("data/core4d/motions")),
    result_root: Path = typer.Option(Path("outputs/balanced")),
    g1_xml: Path = typer.Option(
        Path("third_party/unitree_ros/robots/g1_description/g1_29dof_rev_1_0.xml"),
        exists=True,
        dir_okay=False,
    ),
    output_root: Path = typer.Option(Path("outputs/inverse_dynamics")),
    upsample: int = typer.Option(1, min=1),
    acceleration: str = typer.Option("full", help="full, joints, or none."),
) -> None:
    """Report unconstrained inverse-dynamics torque-limit violations."""
    from .contact import resolve_object_model
    from .mujoco_scene import build_dual_g1_scene_xml
    from .pd_tracking import inverse_dynamics_feasibility

    sequence_dir = motion_root.joinpath(*sequence_id.split("/"))
    object_model = resolve_object_model(sequence_dir, object_models_root)
    stem = sequence_id.replace("/", "_")
    scene_xml = Path("outputs/mujoco") / stem / "dual_g1_pd.xml"
    build_dual_g1_scene_xml(g1_xml, object_model, scene_xml)
    result_dir = result_root / stem
    report = inverse_dynamics_feasibility(
        scene_xml,
        result_dir / "robot1.npz",
        result_dir / "robot2.npz",
        upsample=upsample,
        acceleration=acceleration,
    )
    report["sequence_id"] = sequence_id
    label = stem
    if upsample != 1:
        label += f"_up{upsample}"
    if acceleration != "full":
        label += f"_{acceleration}"
    output_dir = output_root / label
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "metrics.json"
    path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    console.print(json.dumps(report, indent=2))
    console.print(f"Wrote {path}")


@app.command("contact-id")
def contact_id_command(
    sequence_id: str = typer.Argument(help="CORE4D sequence ID."),
    object_models_root: Path = typer.Option(Path("data/core4d/object_models")),
    motion_root: Path = typer.Option(Path("data/core4d/motions")),
    result_root: Path = typer.Option(Path("outputs/balanced_smooth")),
    g1_xml: Path = typer.Option(
        Path("third_party/unitree_ros/robots/g1_description/g1_29dof_rev_1_0.xml"),
        exists=True,
        dir_okay=False,
    ),
    output_root: Path = typer.Option(Path("outputs/contact_id")),
    upsample: int = typer.Option(1, min=1),
    acceleration: str = typer.Option("none"),
    mu: float = typer.Option(0.8, min=0.0),
) -> None:
    """Compare unconstrained ID with two-foot contact-force ID."""
    from .contact import resolve_object_model
    from .contact_id import contact_constrained_id, save_contact_id
    from .mujoco_scene import build_dual_g1_scene_xml

    sequence_dir = motion_root.joinpath(*sequence_id.split("/"))
    object_model = resolve_object_model(sequence_dir, object_models_root)
    stem = sequence_id.replace("/", "_")
    scene_xml = Path("outputs/mujoco") / stem / "dual_g1_pd.xml"
    build_dual_g1_scene_xml(g1_xml, object_model, scene_xml)
    result_dir = result_root / stem
    report = contact_constrained_id(
        scene_xml,
        result_dir / "robot1.npz",
        result_dir / "robot2.npz",
        upsample=upsample,
        acceleration=acceleration,
        mu=mu,
    )
    report["sequence_id"] = sequence_id
    label = f"{stem}_{acceleration}"
    if upsample != 1:
        label += f"_up{upsample}"
    path = output_root / label / "metrics.json"
    save_contact_id(report, path)
    console.print(json.dumps(report, indent=2))
    console.print(f"Wrote {path}")


@app.command("retarget-equilibrium")
def retarget_equilibrium_command(
    sequence_id: str = typer.Argument(help="CORE4D sequence ID."),
    result_root: Path = typer.Option(Path("outputs/balanced_smooth")),
    object_models_root: Path = typer.Option(Path("data/core4d/object_models")),
    motion_root: Path = typer.Option(Path("data/core4d/motions")),
    g1_xml: Path = typer.Option(
        Path("third_party/unitree_ros/robots/g1_description/g1_29dof_rev_1_0.xml"),
        exists=True,
        dir_okay=False,
    ),
    urdf_path: Path = typer.Option(
        Path("third_party/unitree_ros/robots/g1_description/g1_29dof_rev_1_0.urdf"),
        exists=True,
        dir_okay=False,
    ),
    output_root: Path = typer.Option(Path("outputs/equilibrium")),
    frame: int = typer.Option(0, min=0),
    max_iterations: int = typer.Option(80, min=1),
    rollout: bool = typer.Option(False, "--rollout"),
) -> None:
    """Adjust root/legs for unilateral equilibrium, or a short PD-hold rollout."""
    from .contact import resolve_object_model
    from .mujoco_scene import build_dual_g1_scene_xml
    from .retarget.equilibrium import (
        EquilibriumConfig,
        refine_frame_equilibrium,
        refine_frame_rollout,
        save_equilibrium_outputs,
    )

    sequence_dir = motion_root.joinpath(*sequence_id.split("/"))
    object_model = resolve_object_model(sequence_dir, object_models_root)
    stem = sequence_id.replace("/", "_")
    scene_xml = Path("outputs/mujoco") / stem / (
        "dual_g1_pd_feet.xml" if rollout else "dual_g1_pd.xml"
    )
    build_dual_g1_scene_xml(g1_xml, object_model, scene_xml, foot_boxes=rollout)
    result_dir = result_root / stem
    if rollout:
        outputs, diagnostics = refine_frame_rollout(
            scene_xml,
            result_dir / "robot1.npz",
            result_dir / "robot2.npz",
            frame=frame,
            max_iterations=max_iterations,
        )
        if output_root == Path("outputs/equilibrium"):
            output_root = Path("outputs/equilibrium_rollout")
    else:
        outputs, diagnostics = refine_frame_equilibrium(
            scene_xml,
            result_dir / "robot1.npz",
            result_dir / "robot2.npz",
            frame=frame,
            config=EquilibriumConfig(max_iterations=max_iterations),
        )
    diagnostics["sequence_id"] = sequence_id
    output_dir = output_root / stem
    save_equilibrium_outputs(outputs, diagnostics, output_dir, urdf_path)
    console.print(json.dumps(diagnostics, indent=2))
    console.print(f"Wrote {output_dir}")


@app.command("retarget-unilateral")
def retarget_unilateral_command(
    sequence_id: str = typer.Argument(help="CORE4D sequence ID."),
    result_root: Path = typer.Option(Path("outputs/balanced_smooth")),
    object_models_root: Path = typer.Option(Path("data/core4d/object_models")),
    motion_root: Path = typer.Option(Path("data/core4d/motions")),
    g1_xml: Path = typer.Option(
        Path("third_party/unitree_ros/robots/g1_description/g1_29dof_rev_1_0.xml"),
        exists=True,
        dir_okay=False,
    ),
    urdf_path: Path = typer.Option(
        Path("third_party/unitree_ros/robots/g1_description/g1_29dof_rev_1_0.urdf"),
        exists=True,
        dir_okay=False,
    ),
    output_root: Path = typer.Option(Path("outputs/unilateral")),
    start_frame: int = typer.Option(0, min=0),
    max_frames: int | None = typer.Option(None, min=1),
    max_iterations: int | None = typer.Option(None, min=1),
) -> None:
    """Whole-clip unilateral post-pass. Writes a new root; never overwrites balanced_smooth."""
    from .contact import resolve_object_model
    from .mujoco_scene import build_dual_g1_scene_xml
    from .retarget.equilibrium import (
        CLIP_EQUILIBRIUM_CONFIG,
        EquilibriumConfig,
        refine_clip_equilibrium,
        save_equilibrium_outputs,
    )

    if output_root.resolve() == result_root.resolve():
        raise typer.BadParameter("refusing to overwrite result_root; pass a new --output-root")
    if output_root.resolve() == Path("outputs/balanced_smooth").resolve():
        raise typer.BadParameter("refusing to overwrite outputs/balanced_smooth")

    sequence_dir = motion_root.joinpath(*sequence_id.split("/"))
    object_model = resolve_object_model(sequence_dir, object_models_root)
    stem = sequence_id.replace("/", "_")
    scene_xml = Path("outputs/mujoco") / stem / "dual_g1_pd.xml"
    build_dual_g1_scene_xml(g1_xml, object_model, scene_xml)
    result_dir = result_root / stem
    config = CLIP_EQUILIBRIUM_CONFIG
    if max_iterations is not None:
        config = EquilibriumConfig(
            **{**config.__dict__, "max_iterations": max_iterations}
        )
    outputs, diagnostics = refine_clip_equilibrium(
        scene_xml,
        result_dir / "robot1.npz",
        result_dir / "robot2.npz",
        config=config,
        max_frames=max_frames,
        start_frame=start_frame,
    )
    diagnostics["sequence_id"] = sequence_id
    diagnostics["source_root"] = str(result_root)
    output_dir = output_root / stem
    save_equilibrium_outputs(outputs, diagnostics, output_dir, urdf_path)
    summary = {key: value for key, value in diagnostics.items() if key != "frames_detail"}
    console.print(json.dumps(summary, indent=2))
    console.print(f"Wrote {output_dir}")
    console.print(
        "Score with: python -m core4d_dual_retarget.cli contact-id "
        f"{sequence_id} --result-root {output_root} --acceleration none"
    )


@app.command("retarget-stance")
def retarget_stance_command(
    sequence_id: str = typer.Argument(help="CORE4D sequence ID."),
    result_root: Path = typer.Option(Path("outputs/equilibrium")),
    object_models_root: Path = typer.Option(Path("data/core4d/object_models")),
    motion_root: Path = typer.Option(Path("data/core4d/motions")),
    g1_xml: Path = typer.Option(
        Path("third_party/unitree_ros/robots/g1_description/g1_29dof_rev_1_0.xml"),
        exists=True,
        dir_okay=False,
    ),
    urdf_path: Path = typer.Option(
        Path("third_party/unitree_ros/robots/g1_description/g1_29dof_rev_1_0.urdf"),
        exists=True,
        dir_okay=False,
    ),
    output_root: Path = typer.Option(Path("outputs/stance")),
    frame: int = typer.Option(0, min=0),
    all_frames: bool = typer.Option(False, "--all-frames"),
    keep_pose: bool = typer.Option(False, "--keep-pose"),
) -> None:
    """Physics stance post-pass: lift, CoM projection, ankle-pitch hold search."""
    from .contact import resolve_object_model
    from .mujoco_scene import build_dual_g1_scene_xml
    from .retarget.equilibrium import (
        apply_stance_clip,
        refine_stance_pose,
        save_equilibrium_outputs,
    )

    sequence_dir = motion_root.joinpath(*sequence_id.split("/"))
    object_model = resolve_object_model(sequence_dir, object_models_root)
    stem = sequence_id.replace("/", "_")
    scene_xml = Path("outputs/mujoco") / stem / "dual_g1_pd_stance.xml"
    build_dual_g1_scene_xml(
        g1_xml,
        object_model,
        scene_xml,
        foot_boxes=True,
        position_actuators=True,
        stiff_contacts=True,
        leg_kp=200.0,
        arm_kp=40.0,
    )
    result_dir = result_root / stem
    outputs, diagnostics = refine_stance_pose(
        scene_xml,
        result_dir / "robot1.npz",
        result_dir / "robot2.npz",
        frame=frame,
        keep_pose=keep_pose,
    )
    diagnostics["sequence_id"] = sequence_id
    output_dir = output_root / stem
    save_equilibrium_outputs(outputs, diagnostics, output_dir, urdf_path)
    if all_frames:
        outputs, clip_info = apply_stance_clip(
            scene_xml, output_dir / "robot1.npz", output_dir / "robot2.npz"
        )
        diagnostics.update(clip_info)
        save_equilibrium_outputs(outputs, diagnostics, output_dir, urdf_path)
    console.print(json.dumps(diagnostics, indent=2))
    console.print(f"Wrote {output_dir}")


@app.command("filter-swing-com")
def filter_swing_com_command(
    sequence_id: str = typer.Argument(help="CORE4D sequence ID."),
    result_root: Path = typer.Option(Path("outputs/stance")),
    output_root: Path = typer.Option(Path("outputs/stance_comhold")),
    motion_root: Path = typer.Option(Path("data/core4d/motions")),
    object_models_root: Path = typer.Option(Path("data/core4d/object_models")),
    g1_xml: Path = typer.Option(
        Path("third_party/unitree_ros/robots/g1_description/g1_29dof_rev_1_0.xml"),
        exists=True,
        dir_okay=False,
    ),
    person: int = typer.Option(2, "--person", min=1, max=2),
    max_joint_step: float = typer.Option(0.08, "--max-joint-step", min=0.0),
) -> None:
    """Copy a stance clip and rewrite the hover leg so CoM xy stays at frame 0."""
    import shutil

    from .contact import resolve_object_model
    from .mujoco_scene import build_dual_g1_scene_xml
    from .retarget.equilibrium import filter_swing_com_hold

    sequence_dir = motion_root.joinpath(*sequence_id.split("/"))
    object_model = resolve_object_model(sequence_dir, object_models_root)
    stem = sequence_id.replace("/", "_")
    source_dir = result_root / stem
    output_dir = output_root / stem
    if output_dir.resolve() == source_dir.resolve():
        raise typer.BadParameter("output-root must differ from result-root")
    scene_xml = Path("outputs/mujoco") / stem / "dual_g1_pd_stance.xml"
    build_dual_g1_scene_xml(
        g1_xml,
        object_model,
        scene_xml,
        foot_boxes=True,
        position_actuators=True,
        stiff_contacts=True,
        leg_kp=200.0,
        arm_kp=40.0,
    )
    outputs, diagnostics = filter_swing_com_hold(
        scene_xml,
        source_dir / "robot1.npz",
        source_dir / "robot2.npz",
        person=person,
        max_joint_step=max_joint_step,
    )
    diagnostics["sequence_id"] = sequence_id
    diagnostics["source_dir"] = str(source_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    other = 1 if person == 2 else 2
    shutil.copy2(source_dir / f"robot{other}.npz", output_dir / f"robot{other}.npz")
    np.savez_compressed(output_dir / f"robot{person}.npz", **outputs[person])
    (output_dir / "filter_swing_com.json").write_text(
        json.dumps(diagnostics, indent=2) + "\n", encoding="utf-8"
    )
    if (source_dir / "equilibrium.json").exists():
        shutil.copy2(source_dir / "equilibrium.json", output_dir / "equilibrium.json")
    console.print(json.dumps(diagnostics, indent=2))
    console.print(f"Wrote {output_dir}")


@app.command("synthesize-hover")
def synthesize_hover_command(
    sequence_id: str = typer.Argument(help="CORE4D sequence ID."),
    result_root: Path = typer.Option(Path("outputs/stance")),
    output_root: Path = typer.Option(Path("outputs/stance_hover")),
    motion_root: Path = typer.Option(Path("data/core4d/motions")),
    object_models_root: Path = typer.Option(Path("data/core4d/object_models")),
    g1_xml: Path = typer.Option(
        Path("third_party/unitree_ros/robots/g1_description/g1_29dof_rev_1_0.xml"),
        exists=True,
        dir_okay=False,
    ),
    person: int = typer.Option(2, "--person", min=1, max=2),
    yaw_amp: float = typer.Option(0.12, "--yaw-amp"),
    roll_amp: float = typer.Option(0.0, "--roll-amp"),
    period_s: float = typer.Option(2.0, "--period", min=0.1),
    hold_com: bool = typer.Option(True, "--hold-com/--no-hold-com"),
) -> None:
    """Copy a stance clip and replace the hover leg with a CoM-neutral hip sine."""
    import shutil

    from .contact import resolve_object_model
    from .mujoco_scene import build_dual_g1_scene_xml
    from .retarget.equilibrium import synthesize_hover

    sequence_dir = motion_root.joinpath(*sequence_id.split("/"))
    object_model = resolve_object_model(sequence_dir, object_models_root)
    stem = sequence_id.replace("/", "_")
    source_dir = result_root / stem
    output_dir = output_root / stem
    if output_dir.resolve() == source_dir.resolve():
        raise typer.BadParameter("output-root must differ from result-root")
    scene_xml = Path("outputs/mujoco") / stem / "dual_g1_pd_stance.xml"
    build_dual_g1_scene_xml(
        g1_xml,
        object_model,
        scene_xml,
        foot_boxes=True,
        position_actuators=True,
        stiff_contacts=True,
        leg_kp=200.0,
        arm_kp=40.0,
    )
    outputs, diagnostics = synthesize_hover(
        scene_xml,
        source_dir / "robot1.npz",
        source_dir / "robot2.npz",
        person=person,
        yaw_amp=yaw_amp,
        roll_amp=roll_amp,
        period_s=period_s,
        hold_com=hold_com,
    )
    diagnostics["sequence_id"] = sequence_id
    output_dir.mkdir(parents=True, exist_ok=True)
    other = 1 if person == 2 else 2
    shutil.copy2(source_dir / f"robot{other}.npz", output_dir / f"robot{other}.npz")
    np.savez_compressed(output_dir / f"robot{person}.npz", **outputs[person])
    (output_dir / "synthesize_hover.json").write_text(
        json.dumps(diagnostics, indent=2) + "\n", encoding="utf-8"
    )
    console.print(json.dumps(diagnostics, indent=2))
    console.print(f"Wrote {output_dir}")


@app.command("pack-dataset")
def pack_dataset_command(
    result_root: Path = typer.Option(Path("outputs/balanced_smooth")),
    metrics_root: Path = typer.Option(Path("outputs/metrics/balanced_smooth")),
    output_path: Path = typer.Option(Path("outputs/dataset/manifest.json")),
    method: str = typer.Option("balanced_smooth"),
    generalization: bool = typer.Option(False, "--generalization"),
    geometry_gate: bool = typer.Option(
        False,
        "--geometry-gate",
        help="Use sole/hand grid publish bar instead of the legacy ankle/palm-5cm gate.",
    ),
) -> None:
    """Write quality labels and a sequence-level train/val/test split."""
    from .dataset import GENERALIZATION_SEQUENCES, build_dataset_manifest

    manifest = build_dataset_manifest(
        result_root,
        metrics_root,
        output_path,
        method=method,
        sequence_ids=GENERALIZATION_SEQUENCES if generalization else None,
        geometry_gate=geometry_gate,
    )
    console.print(json.dumps(manifest, indent=2))
    console.print(f"Wrote {output_path}")


@app.command("layer2-report")
def layer2_report_command(
    baseline_id_root: Path = typer.Option(Path("outputs/contact_id")),
    new_id_root: Path = typer.Option(Path("outputs/contact_id_unilateral_v3_s1")),
    baseline_metrics_root: Path = typer.Option(Path("outputs/metrics/balanced_smooth")),
    new_metrics_root: Path = typer.Option(Path("outputs/metrics/unilateral_v3_s1")),
    output_path: Path = typer.Option(Path("outputs/unilateral_v3_s1/layer2_report.json")),
) -> None:
    """Compare unilateral post-pass vs kinematic release on box clips."""
    from .dataset import build_layer2_report

    report = build_layer2_report(
        baseline_id_root,
        new_id_root,
        baseline_metrics_root,
        new_metrics_root,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    table = Table(title="Layer 2: geometry + unilateral ID")
    table.add_column("clip")
    table.add_column("geom")
    table.add_column("tensile r1")
    table.add_column("tensile r2")
    table.add_column("uni residual r1")
    table.add_column("uni residual r2")
    table.add_column("palm cm r1")
    table.add_column("jerk r1")
    for row in report["sequences"]:
        r1 = row["robots"]["1"]
        r2 = row["robots"]["2"]
        table.add_row(
            row["sequence_id"],
            "pass" if row["passed"] else ",".join(
                r1["geometry_failures"] + [f"r2:{f}" for f in r2["geometry_failures"]]
            ) or "fail",
            f"{r1['tensile_baseline']:.2f}->{r1['tensile_new']:.2f}",
            f"{r2['tensile_baseline']:.2f}->{r2['tensile_new']:.2f}",
            f"{r1['unilateral_residual_baseline']:.0f}->{r1['unilateral_residual_new']:.0f}",
            f"{r2['unilateral_residual_baseline']:.0f}->{r2['unilateral_residual_new']:.0f}",
            f"{r1['palm_cm_new']:.1f}",
            f"{r1['jerk_new']:.1f}",
        )
    console.print(table)
    console.print(
        "mean tensile "
        f"{report['mean_tensile_baseline']:.3f} -> {report['mean_tensile_new']:.3f}; "
        "mean uni residual "
        f"{report['mean_unilateral_residual_baseline']:.0f} -> "
        f"{report['mean_unilateral_residual_new']:.0f}"
    )
    if report["missing"]:
        console.print(f"missing: {report['missing']}")
    console.print(f"Wrote {output_path}")


@app.command("train-residual-stance")
def train_residual_stance_command(
    sequence_id: str = typer.Argument(help="CORE4D sequence ID."),
    motion_root: Path = typer.Option(Path("data/core4d/motions")),
    object_models_root: Path = typer.Option(Path("data/core4d/object_models")),
    result_root: Path = typer.Option(Path("outputs/stance")),
    g1_xml: Path = typer.Option(
        Path("third_party/unitree_ros/robots/g1_description/g1_29dof_rev_1_0.xml"),
        exists=True,
        dir_okay=False,
    ),
    output_path: Path = typer.Option(Path("outputs/residual_policy/gains.json")),
    duration: float = typer.Option(6.0, min=1.0),
    samples: int = typer.Option(18, min=4),
    residual_joints: str = typer.Option("ankle", "--residual-joints"),
) -> None:
    """Fit a 4-gain IMU residual so the stance pose survives a long hold."""
    from .contact import resolve_object_model
    from .mujoco_scene import build_dual_g1_scene_xml
    from .residual import train_linear_stance_residual

    sequence_dir = motion_root.joinpath(*sequence_id.split("/"))
    object_pose_path = sequence_dir / "smooth_objposes_so3.npy"
    if not object_pose_path.exists():
        object_pose_path = sequence_dir / "smooth_objposes.npy"
    object_model = resolve_object_model(sequence_dir, object_models_root)
    stem = sequence_id.replace("/", "_")
    scene_xml = Path("outputs/mujoco") / stem / "dual_g1_pd_stance.xml"
    build_dual_g1_scene_xml(
        g1_xml,
        object_model,
        scene_xml,
        foot_boxes=True,
        position_actuators=True,
        stiff_contacts=True,
        leg_kp=200.0,
        arm_kp=40.0,
    )
    result_dir = result_root / stem
    report = train_linear_stance_residual(
        scene_xml,
        np.load(object_pose_path),
        result_dir / "robot1.npz",
        result_dir / "robot2.npz",
        duration=duration,
        samples=samples,
        residual_joints=residual_joints,
    )
    report["sequence_id"] = sequence_id
    report["residual_joints"] = residual_joints
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    console.print(json.dumps(report, indent=2))
    console.print(f"Wrote {output_path}")


@app.command("export-residual-obs")
def export_residual_obs_command(
    motion_root: Path = typer.Option(Path("data/core4d/motions")),
    result_root: Path = typer.Option(Path("outputs/balanced_smooth")),
    output_root: Path = typer.Option(Path("outputs/residual_obs")),
    include_holdout: bool = typer.Option(True, "--include-holdout/--box-only"),
) -> None:
    """Pack kinematic observations for residual RL or WBC tracking."""
    from .residual import export_residual_dataset

    manifest = export_residual_dataset(
        motion_root,
        result_root,
        output_root,
        include_holdout=include_holdout,
    )
    console.print(json.dumps(manifest, indent=2))
    console.print(f"Wrote {output_root / 'manifest.json'}")


if __name__ == "__main__":
    app()
