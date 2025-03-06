# CLAUDE.md: Development Guidelines

## Build & Test Commands
- Install: `make local-install`
- Lint: `make lint`
- Format: `make format`
- Typecheck: `make typecheck`
- Run tests: `make mactest` or `pytest -m 'not envpool and not slow'`
- Run single test: `pytest tests/test_file.py::test_function -v`
- Training command: `python -m cleanba.cleanba_impala --from-py-fn=cleanba.config:sokoban_drc33_59`

## Code Style Guidelines
- **Formatting**: Follow ruff format (127 line length)
- **Imports**: Use isort through ruff, with known third-party libraries like wandb
- **Types**: Use type annotations, checked with pyright
- **Naming**: 
  - Variables: `snake_case`
  - Classes: `PascalCase`
  - Constants: `UPPER_CASE`
- **Structure**: Modules organized by functionality (cleanba, experiments, tests)
- **Error handling**: Use asserts for validation in tests, exceptions for runtime errors
- **Documentation**: Include docstrings for public functions and classes
- **JAX/Flax patterns**: Use pure functions and maintain functional style

## Key Files
- **environments.py**: Contains environment wrappers and config classes for different environments (Sokoban, Boxoban, Craftax). Includes `EpisodeEvalWrapper` for logging episode returns and adapters for different environment backends.

- **cleanba_impala.py**: Main training loop implementation for the IMPALA algorithm. Contains multi-threaded rollout data collection, parameter synchronization, and training. Uses `WandbWriter` for logging, queues for communication between rollout and learner threads, and implements checkpointing.

- **impala_loss.py**: Implements V-trace and PPO loss functions. Contains `Rollout` data structure, TD-error computation with V-trace, and policy gradient calculations. Handles truncated episodes specially to provide correct advantage estimates.