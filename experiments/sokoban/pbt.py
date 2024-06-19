from pathlib import Path

from cleanba.launcher import FlamingoRun, group_from_fname, launch_jobs

num_samples = 4
run = FlamingoRun(
    [["python", "-m", "cleanba.pbt", f"-n={num_samples}"]],
    CONTAINER_TAG="62d97f2-main",
    CPU=4 * num_samples,
    MEMORY=f"{15 * num_samples}G",
    SHM_SIZE=f"{6 * num_samples}G",
    GPU=1 * num_samples,
    PRIORITY="normal-batch",
    XLA_PYTHON_CLIENT_MEM_FRACTION='".95"',
)


GROUP: str = group_from_fname(__file__)

if __name__ == "__main__":
    launch_jobs(
        [run],
        group=GROUP,
        job_template_path=Path(__file__).parent.parent.parent / "k8s/runner.yaml",
        project="cleanba",
    )
