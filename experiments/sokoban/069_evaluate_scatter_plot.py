from pathlib import Path

from farconf import update_fns_to_cli

from cleanba.launcher import FlamingoRun, group_from_fname, launch_jobs
from cleanba.load_and_eval import LoadAndEvalArgs, default_load_and_eval

group_to_subdir = {
    "/training/cleanba/061-pfinal2/wandb": [
        "run-20240618_205932-syb50iz7",
        "run-20240618_210240-28n07cac",
        "run-20240618_210241-qqp0kn15",
        "run-20240618_205934-bkynosqi",
        "run-20240618_205933-q4mjldyy",
        "run-20240618_205941-zgyp3v0o",
        "run-20240618_205935-13qckf6e",
        "run-20240618_205940-8ul1b23e",
        "run-20240618_205941-gobfm3wm",
        "run-20240618_205934-jl6bq8ih",
    ],
    "/training/cleanba/061-pfinal2-drc11/wandb": [
        "run-20240623_041338-3i5nocf6",
        "run-20240623_041344-v2fm2qze",
        "run-20240623_041342-nom9jda6",
        "run-20240623_041343-3a2pv9yr",
        "run-20240623_041343-eue6pax7",
    ],
    "/training/cleanba/064-ablate-back-fixed/wandb": [
        "run-20240627_064349-m7mai9i8",
        "run-20240627_064413-jasi92bn",
        "run-20240627_064416-jo2nnr0k",
        "run-20240627_064413-idu01sdd",
        "run-20240627_064415-vn0zt36i",
        "run-20240627_064414-afl5t1jz",
        "run-20240627_064352-yg05sh6c",
        "run-20240627_064402-6r7s6u21",
        "run-20240627_064404-56rxj5an",
        "run-20240627_064401-u9zeuq1b",
        "run-20240627_064402-vpcqc6lj",
        "run-20240627_064413-4ayrkwe1",
        "run-20240627_064351-yjzplhr1",
        "run-20240627_064408-kfhwjowl",
        "run-20240627_064413-robdj0r9",
        "run-20240627_064413-fuougei1",
        "run-20240627_064402-f5osnuxc",
        "run-20240627_064415-gzt5aoao",
        "run-20240627_064402-2pxyq7pe",
        "run-20240627_064404-jh3h6nk8",
        "run-20240627_064407-kgjcqbpw",
        "run-20240627_064414-xaolan0s",
        "run-20240627_064405-lzazikkc",
        "run-20240627_064413-m3cs1npf",
    ],
    "/training/cleanba/064-ablate-back/wandb": [
        "run-20240626_120809-yr6do0ok",
        "run-20240626_211636-yclelqia",
        "run-20240626_154851-55ey3fe3",
        "run-20240626_123059-0nec5vv9",
        "run-20240626_211631-4ka49qcg",
        "run-20240626_211636-kig9w1k6",
        "run-20240626_082437-mxezj419",
        "run-20240626_082437-49zw4m3g",
        "run-20240627_020641-j4nj4taa",
        "run-20240626_211636-y9e5e682",
        "run-20240627_020643-gpl1d64b",
        "run-20240626_211640-smu1zwej",
        "run-20240626_153458-2rlqdr5c",
        "run-20240626_155120-nqroy8ks",
        "run-20240626_154905-p549yoh9",
        "run-20240626_155007-8nbg3jqe",
        "run-20240626_152607-4kexyqew",
        "run-20240626_153459-rhqbpcxb",
        "run-20240626_082417-nt25b2jg",
        "run-20240626_155656-3p3ztmva",
        "run-20240626_120531-6rgxmgr8",
        "run-20240626_120909-59wybefh",
        "run-20240626_211646-hljo6n14",
        "run-20240626_211643-3nm1b3ys",
        "run-20240626_211633-e7xpv0t9",
        "run-20240626_120612-cdlleqr1",
        "run-20240627_020641-cptzoqjt",
        "run-20240626_211637-kgj9zene",
        "run-20240626_211635-t3p312lr",
        "run-20240626_230753-0z01y4qe",
        "run-20240626_114033-h7eqkpop",
        "run-20240626_211635-xec6frqv",
        "run-20240626_211654-4znyk009",
        "run-20240626_153501-1alsk9oe",
        "run-20240626_144616-crv0f89c",
        "run-20240626_153500-ubvdi939",
        "run-20240626_154958-9k6msid4",
        "run-20240626_211633-psqylhm6",
        "run-20240626_145037-vmko3gtx",
        "run-20240626_131506-ujsk6uqo",
        "run-20240626_153503-4h038ciq",
        "run-20240626_154955-bbl7763z",
        "run-20240626_153450-dh3egbev",
        "run-20240626_211701-9n690m24",
        "run-20240626_114016-zx2p2o1g",
        "run-20240626_153501-yw0s4lp6",
        "run-20240626_153458-6vl4ozzu",
        "run-20240626_211643-6tzv00le",
        "run-20240626_125022-z3jeat5e",
        "run-20240626_230753-uyj3x15e",
        "run-20240626_153503-nsq8e8yg",
        "run-20240626_160402-jxkki05l",
        "run-20240626_153457-v1n55rpl",
        "run-20240626_114034-gcp8vsgy",
        "run-20240626_154943-252oepjl",
        "run-20240626_114315-tq078cjk",
        "run-20240626_114316-2d1epob7",
        "run-20240626_082417-4ptrrzwi",
        "run-20240626_230751-tjblfwv4",
        "run-20240626_153449-p1cxaoku",
        "run-20240626_155001-3bitm9nu",
        "run-20240626_211632-igwodlho",
        "run-20240626_211634-qlcpg4t1",
        "run-20240626_122615-vtlw960a",
        "run-20240626_114334-1naw70o5",
        "run-20240626_153459-zfmtq8hv",
        "run-20240626_151328-19gx02cv",
        "run-20240627_010444-narse2uo",
        "run-20240626_211645-xvyvfflr",
        "run-20240626_211643-wq0srea7",
        "run-20240626_211632-6nyqqtqc",
        "run-20240627_010444-b38z68c3",
        "run-20240627_010445-9fgcybke",
        "run-20240627_010444-cdsg92aa",
        "run-20240626_211637-1d6jq3uo",
        "run-20240626_211634-7wzo7tqn",
        "run-20240626_123155-1zz5ttlf",
        "run-20240626_154941-n5vq7gda",
        "run-20240626_155019-kx0h1axm",
        "run-20240626_120602-eql283jg",
        "run-20240626_114033-znbc4rz4",
        "run-20240626_154939-g7clt7mu",
        "run-20240626_153457-kxyjdn28",
        "run-20240626_211632-bpip5wdr",
        "run-20240626_114033-okjscwoe",
        "run-20240627_020640-rtab9pvw",
        "run-20240626_153500-bdbt5m7j",
        "run-20240626_154932-yg5sox60",
        "run-20240626_160305-dqsaq1tb",
        "run-20240626_082435-7la3c32h",
        "run-20240626_211634-1fg88mi7",
        "run-20240626_211633-kcttmt5u",
        "run-20240627_015708-q70lwpqf",
        "run-20240626_154955-xmuqzu55",
        "run-20240626_114525-mb3kvc25",
        "run-20240626_082438-hovrwjgw",
        "run-20240626_114602-6iufrw5c",
        "run-20240626_155656-fxuqx7wl",
        "run-20240626_114327-iwpi0y9h",
        "run-20240626_114033-o2tyul1k",
        "run-20240626_120740-1eqpzsxr",
        "run-20240627_015706-f3gilr54",
        "run-20240626_211634-n2po9vs5",
        "run-20240626_123150-tjx2sror",
        "run-20240626_211632-po7xj6iq",
        "run-20240626_122859-kidkvo5x",
        "run-20240626_114334-r4x8b25u",
        "run-20240626_114334-pl57snnp",
        "run-20240626_082438-f8e2aeu1",
        "run-20240626_153457-nl55d5fn",
        "run-20240626_210440-wkaovk0g",
        "run-20240626_125558-5jdmxpak",
        "run-20240627_015707-ew6mcjox",
        "run-20240627_015705-70d5fj2a",
        "run-20240626_155703-x11a3v75",
        "run-20240626_210139-q7ieg74g",
        "run-20240626_211648-k4cpwm4s",
        "run-20240626_160309-dx9k0779",
        "run-20240626_155003-y4z9wxwy",
        "run-20240626_153500-m364rrjc",
        "run-20240626_160326-1u8j4p16",
        "run-20240626_160401-fqavmnu5",
        "run-20240626_114336-47yhch6e",
        "run-20240626_152313-n7x7v5za",
        "run-20240626_114017-h4i9lfxd",
        "run-20240626_154900-9v4g706b",
    ],
    "/training/cleanba/066-try-with-old-params/wandb": [
        "run-20240628_054537-zwyqbhfi",
        "run-20240628_054527-204i1sft",
        "run-20240628_053822-ykcwwnv5",
        "run-20240628_054535-j8dvnw89",
        "run-20240628_054538-mf0qrhlg",
        "run-20240628_054528-8z3n2egh",
        "run-20240628_053814-ubuiao9u",
        "run-20240628_053813-r0oiocjz",
        "run-20240628_053815-mmxno7kl",
        "run-20240628_054526-azum9ppz",
        "run-20240628_055145-66hhykh8",
        "run-20240628_054527-anu85u0c",
        "run-20240628_054527-yjdb3ibt",
        "run-20240628_053815-u8wypixh",
        "run-20240628_053815-f9dgsdpv",
        "run-20240628_054527-5vqd9j8v",
        "run-20240628_054528-l9az4nd4",
        "run-20240628_054527-pnkw63qd",
    ],
}

runs_to_evaluate = [Path(k) / v for k, vs in group_to_subdir.items() for v in vs]


clis: list[list[str]] = []
for load_path in runs_to_evaluate:

    def update(config: LoadAndEvalArgs) -> LoadAndEvalArgs:
        config.load_other_run = load_path
        config.only_last_checkpoint = True

        env = config.eval_envs.pop("valid_medium")
        env.env.split = "planning"
        env.env.n_levels_to_load = 5000
        env.n_episode_multiple = 10
        config.eval_envs["planning_medium"] = env

        for env in config.eval_envs.values():
            env.env.steps_to_think = [0, 2, 4, 6, 8, 10, 12]
        return config

    cli, _ = update_fns_to_cli(default_load_and_eval, update)
    clis.append(cli)


runs: list[FlamingoRun] = []
RUNS_PER_MACHINE = len(clis) // 5
for update_fns_i in range(0, len(clis), RUNS_PER_MACHINE):
    this_run_clis = [
        ["python", "-m", "cleanba.load_and_eval", *clis[update_fns_i + j]]
        for j in range(min(RUNS_PER_MACHINE, len(clis) - update_fns_i))
    ]
    print(this_run_clis)
    runs.append(
        FlamingoRun(
            this_run_clis,
            CONTAINER_TAG="6f8d92b-main",
            CPU=12,
            MEMORY="40G",
            GPU=1,
            PRIORITY="high-batch",
            XLA_PYTHON_CLIENT_MEM_FRACTION='".45"',
            parallel=False,
        )
    )


GROUP: str = group_from_fname(__file__)

if __name__ == "__main__":
    launch_jobs(
        runs,
        group=GROUP,
        job_template_path=Path(__file__).parent.parent.parent / "k8s/runner.yaml",
        project="cleanba",
    )
