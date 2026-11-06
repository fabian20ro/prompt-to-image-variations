#!/usr/bin/env python3
"""CLI entry point for the FLUX 3 Image Klein prompt generator."""

import shutil
import sys
from pathlib import Path

import click
from config import settings, paths
from grammar_generator import generate_grammar
from image_enhancer import enhance_image, collect_images
from pipeline import PipelineExecutor
from utils import check_lm_studio


def clean_generated():
    """Remove all generated files (grammars and prompts)."""
    count = 0
    for d in [paths.grammars_dir, paths.prompts_dir]:
        if d.exists():
            for item in d.iterdir():
                if item.is_file():
                    item.unlink()
                    count += 1
                elif item.is_dir():
                    shutil.rmtree(item)
                    count += 1
    return count


def cli_progress(stage: str, current: int = 0, total: int = 0, message: str = "") -> None:
    """Progress callback for CLI that uses click.echo."""
    if message:
        if current > 0 and total > 0:
            click.echo(f"  [{current}/{total}] {message}")
        else:
            click.echo(message)


@click.command()
@click.option(
    '-p', '--prompt',
    default=None,
    help='Image description to generate variations for'
)
@click.option(
    '--clean',
    is_flag=True,
    help='Remove all generated files (grammars and prompts)'
)
@click.option(
    '-n', '--count',
    default=50,
    type=click.IntRange(min=1),
    help='Number of variations to generate (default: 50)'
)
@click.option(
    '-o', '--output',
    type=click.Path(path_type=Path),
    help='Output directory (default: generated/prompts/{hash}_{timestamp}/)'
)
@click.option(
    '--dry-run',
    is_flag=True,
    help='Generate and display grammar without generating images'
)
@click.option(
    '--no-cache',
    is_flag=True,
    help='Skip grammar cache and regenerate'
)
@click.option(
    '--base-url',
    default=settings.lm_studio.base_url,
    help='LM Studio API base URL (default: %s)' % settings.lm_studio.base_url,
)
@click.option(
    '--temperature',
    default=0.7,
    type=float,
    help='LLM temperature for grammar generation (default: 0.7)'
)
@click.option(
    '--generate-images', '-i',
    is_flag=True,
    help='Generate images for each prompt using mflux'
)
@click.option(
    '--images-per-prompt',
    default=1,
    type=click.IntRange(min=0),
    help='Number of images to generate per prompt (default: 1, 0 = prompt-only layout)'
)
@click.option(
    '--prefix',
    default=None,
    help='Prefix for output files (default: "image")'
)
@click.option(
    '--width',
    default=None,
    type=int,
    help='Image width in pixels (default: %s)' % settings.image_generation.default_width
)
@click.option(
    '--height',
    default=None,
    type=int,
    help='Image height in pixels (default: %s)' % settings.image_generation.default_height
)
@click.option(
    '--max-prompts',
    type=int,
    default=None,
    help='Limit number of prompts to render images for (default: all)'
)
@click.option(
    '--seed',
    type=int,
    default=None,
    help='Random seed for image generation (default: random)'
)
@click.option(
    '--from-grammar',
    type=click.Path(exists=True, path_type=Path),
    help='Resume from existing grammar file (skip LLM generation)'
)
@click.option(
    '--from-prompts',
    type=click.Path(exists=True, path_type=Path),
    help='Resume from existing prompts directory (images only)'
)
@click.option(
    '--enhance',
    is_flag=True,
    help='Enable SeedVR2 2x enhancement after image generation'
)
@click.option(
    '--enhance-softness',
    default=0.5,
    type=float,
    help='Enhancement softness (0.0-1.0, default: 0.5)'
)
@click.option(
    '--enhance-after',
    is_flag=True,
    help='Defer enhancement to after all images are generated (saves memory)'
)
@click.option(
    '--enhance-images',
    type=str,
    help='Standalone: enhance existing images from file, folder, or glob pattern'
)
@click.option(
    '--resume',
    is_flag=True,
    help='Skip already-generated images when resuming interrupted runs'
)
@click.option(
    '--no-enhance-tiled-vae',
    is_flag=True,
    help='Disable SeedVR2 tiled VAE decoding; does not affect FLUX generation'
)
@click.option(
    '--serve',
    is_flag=True,
    help='Start the web UI server at http://localhost:8000'
)
@click.option(
    '--port',
    default=8000,
    type=int,
    help='Port for web UI server (default: 8000)'
)
def main(
    prompt: str | None,
    clean: bool,
    count: int,
    output: Path | None,
    dry_run: bool,
    no_cache: bool,
    base_url: str,
    temperature: float,
    generate_images: bool,
    images_per_prompt: int,
    prefix: str | None,
    width: int | None,
    height: int | None,
    max_prompts: int | None,
    seed: int | None,
    from_grammar: Path | None,
    from_prompts: Path | None,
    enhance: bool,
    enhance_softness: float,
    enhance_after: bool,
    enhance_images: str | None,
    resume: bool,
    no_enhance_tiled_vae: bool,
    serve: bool,
    port: int,
):
    """
    Generate FLUX 3 Image Klein prompt variations using local LLM-powered Tracery grammars.

    Example:
        uv run python src/cli.py -p "a dragon flying over mountains" -n 50
        uv run python src/cli.py --clean  # Remove all generated files

    With image generation:
        uv run python src/cli.py -p "a dragon flying over mountains" -n 5 \
            --generate-images --images-per-prompt 3 --prefix dragon

    With image generation + SeedVR2 2x enhancement:
        uv run python src/cli.py -p "a cat" -n 1 --generate-images --enhance --prefix test

    Resume from existing grammar:
        uv run python src/cli.py --from-grammar generated/grammars/abc123.tracery.json -n 100

    Resume from existing prompts (images only):
        uv run python src/cli.py --from-prompts generated/prompts/abc123_20260124_122208 \
            --generate-images --images-per-prompt 2

    Standalone enhancement (no image generation):
        uv run python src/cli.py --enhance-images path/to/image.png
        uv run python src/cli.py --enhance-images path/to/folder/
        uv run python src/cli.py --enhance-images "generated/prompts/*/test_*.png"
    """
    # LM Studio is required only when creating a grammar.
    if prompt and not from_grammar and not from_prompts and not check_lm_studio(base_url):
        click.echo(f"Error: LM Studio is not reachable at {base_url}. Please ensure it is running.", err=True)
        sys.exit(1)

    # Handle --clean
    if clean:
        removed = clean_generated()
        click.echo(f"Cleaned {removed} items from {paths.generated_dir}")
        if not prompt and not from_grammar and not from_prompts and not enhance_images and not serve:
            return

    # Handle --serve: Start web UI server
    if serve:
        import webbrowser
        import threading

        click.echo(f"Starting web UI server at http://localhost:{port}")
        click.echo("Press Ctrl+C to stop")

        def open_browser():
            import time
            time.sleep(1.5)
            webbrowser.open(f"http://localhost:{port}")

        threading.Thread(target=open_browser, daemon=True).start()

        try:
            import uvicorn
            from server.app import app
            uvicorn.run(app, host="0.0.0.0", port=port, log_level="info")
        except ImportError:
            click.echo("Error: Missing dependencies for web UI. Install with: uv sync", err=True)
            sys.exit(1)
        return

    # STANDALONE MODE: Enhance existing images
    if enhance_images:
        _run_standalone_enhancement(enhance_images, enhance_softness, seed, no_enhance_tiled_vae)
        return

    # Validation
    if from_grammar and from_prompts:
        click.echo("Error: Cannot use both --from-grammar and --from-prompts", err=True)
        sys.exit(1)

    if from_prompts and not generate_images:
        click.echo("Error: --from-prompts requires --generate-images", err=True)
        sys.exit(1)

    if not prompt and not from_grammar and not from_prompts and not clean and not enhance_images:
        click.echo("Error: --prompt is required (or use --from-grammar/--from-prompts/--enhance-images/--serve)", err=True)
        sys.exit(1)

    if prompt and (from_grammar or from_prompts):
        click.echo("Warning: --prompt is ignored when using --from-grammar or --from-prompts", err=True)

    if not (0.0 <= temperature <= 2.0):
        raise click.BadParameter(f"temperature must be between 0.0 and 2.0, got {temperature}")

    # Set defaults
    prefix = prefix or "image"
    if not from_prompts:
        width = settings.image_generation.default_width if width is None else width
        height = settings.image_generation.default_height if height is None else height
    from image_generator import validate_dimensions
    try:
        validate_dimensions(width if width is not None else 864, height if height is not None else 1152)
    except ValueError as exc:
        raise click.BadParameter(str(exc)) from exc

    # Handle --dry-run (needs special handling for grammar preview)
    if dry_run:
        _run_dry_run(prompt, from_grammar, base_url, no_cache, temperature, width, height)
        return

    # Create executor with CLI progress callback
    executor = PipelineExecutor(on_progress=cli_progress)

    # MODE C: From prompts (images only)
    if from_prompts:
        click.echo(f"Loading prompts from: {from_prompts}")
        result = executor.run_from_prompts(
            prompts_dir=from_prompts,
            images_per_prompt=images_per_prompt,
            width=width,
            height=height,
            seed=seed,
            max_prompts=max_prompts,
            enhance_tiled_vae=not no_enhance_tiled_vae,
            enhance=enhance,
            enhance_softness=enhance_softness,
            enhance_after=enhance_after,
            resume=resume,
        )

    # MODE B: From grammar (skip LLM generation)
    elif from_grammar:
        click.echo(f"Loading grammar from: {from_grammar}")
        result = executor.run_from_grammar(
            grammar_path=from_grammar,
            count=count,
            prefix=prefix,
            generate_images=generate_images,
            images_per_prompt=images_per_prompt,
            width=width,
            height=height,
            seed=seed,
            max_prompts=max_prompts,
            enhance_tiled_vae=not no_enhance_tiled_vae,
            enhance=enhance,
            enhance_softness=enhance_softness,
            enhance_after=enhance_after,
            resume=resume,
            output_dir=output,
        )

    # MODE A: Full pipeline (default)
    else:
        click.echo(f"Generating grammar for: {prompt}")
        result = executor.run_full_pipeline(
            prompt=prompt,
            base_url=base_url,
            count=count,
            prefix=prefix,
            temperature=temperature,
            no_cache=no_cache,
            generate_images=generate_images,
            images_per_prompt=images_per_prompt,
            width=width,
            height=height,
            seed=seed,
            max_prompts=max_prompts,
            enhance_tiled_vae=not no_enhance_tiled_vae,
            enhance=enhance,
            enhance_softness=enhance_softness,
            enhance_after=enhance_after,
            resume=resume,
            output_dir=output,
        )

    # Handle result
    if not result.success:
        click.echo(f"Error: {result.error}", err=True)
        sys.exit(1)

    # Print summary
    click.echo(f"\nGenerated {result.prompt_count} prompts in: {result.output_dir}")
    if result.image_count > 0:
        if result.skipped_count > 0:
            click.echo(f"Generated {result.image_count} images, skipped {result.skipped_count} existing")
        else:
            click.echo(f"Generated {result.image_count} images")


def _run_standalone_enhancement(
    enhance_images: str,
    enhance_softness: float,
    seed: int | None,
    no_enhance_tiled_vae: bool,
) -> None:
    """Run standalone image enhancement mode."""
    click.echo(f"Collecting images from: {enhance_images}")
    try:
        images = collect_images(enhance_images)
    except ValueError as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)

    click.echo(f"Enhancing {len(images)} images with SeedVR2 (2x upscale, softness={enhance_softness})...")

    current_seed = seed

    for idx, img_path in enumerate(images, 1):
        click.echo(f"  [{idx}/{len(images)}] Enhancing {img_path.name} (replacing original)...")

        try:
            enhance_image(
                image_path=img_path,
                output_path=img_path,
                softness=enhance_softness,
                seed=current_seed,
                tiled_vae=not no_enhance_tiled_vae,
            )
        except ImportError as e:
            click.echo(f"Error: {e}", err=True)
            sys.exit(1)
        except Exception as e:
            click.echo(f"Error enhancing image: {e}", err=True)
            sys.exit(1)

        if current_seed is not None:
            current_seed += 1

    click.echo(f"\nEnhanced {len(images)} images")


def _run_dry_run(
    prompt: str | None,
    from_grammar: Path | None,
    base_url: str,
    no_cache: bool,
    temperature: float,
    width: int = 864,
    height: int = 1152,
) -> None:
    """Run dry-run mode to preview grammar."""
    if from_grammar:
        click.echo(f"Loading grammar from: {from_grammar}")
        grammar = from_grammar.read_text()
        click.echo("\n--- Loaded Grammar ---\n")
        click.echo(grammar)
        click.echo("\n--- End Grammar ---")
    elif prompt:
        click.echo(f"Generating grammar for: {prompt}")
        try:
            grammar, was_cached, _ = generate_grammar(
                user_prompt=prompt,
                base_url=base_url,
                use_cache=not no_cache,
                temperature=temperature,
                width=width,
                height=height,
            )
        except Exception as e:
            click.echo(f"Error generating grammar: {e}", err=True)
            click.echo("Make sure LM Studio is running at " + base_url, err=True)
            sys.exit(1)

        if was_cached:
            click.echo("Using cached grammar")
        else:
            click.echo("Generated new grammar via LM Studio")

        click.echo("\n--- Generated Grammar ---\n")
        click.echo(grammar)
        click.echo("\n--- End Grammar ---")
    else:
        click.echo("Error: --prompt is required for --dry-run", err=True)
        sys.exit(1)


if __name__ == '__main__':
    main()
