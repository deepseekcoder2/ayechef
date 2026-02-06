#!/usr/bin/env python3
"""
Rich Progress UI for Mealie Operations
======================================

Provides beautiful terminal UI with progress bars, timers, and status indicators.
Uses rich library for enhanced terminal experience.
"""

# Clean, minimal branding
MEALIE_HEADER = """
Mealie Recipe Management System
==============================="""

MEALIE_PROMPT = "mealie> "

import time
import threading
from typing import Optional, Callable
from dataclasses import dataclass
from datetime import datetime, timedelta

try:
    from rich.console import Console
    from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn, TimeRemainingColumn, TimeElapsedColumn
    from rich.panel import Panel
    from rich.text import Text
    from rich.table import Table
    from rich.live import Live
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False
    Console = None  # Define for type hints
    print("⚠️ Rich library not available. Install with: pip install rich")
    print("   Falling back to basic progress reporting.")


@dataclass
class ProgressStats:
    """Statistics for progress tracking."""
    total: int
    completed: int = 0
    failed: int = 0
    start_time: float = 0.0
    last_update: float = 0.0

    @property
    def elapsed_time(self) -> float:
        return time.time() - self.start_time

    @property
    def success_rate(self) -> float:
        if self.completed + self.failed == 0:
            return 0.0
        return self.completed / (self.completed + self.failed)

    @property
    def eta_seconds(self) -> Optional[float]:
        if self.completed == 0:
            return None
        avg_time_per_item = self.elapsed_time / self.completed
        remaining_items = self.total - (self.completed + self.failed)
        return avg_time_per_item * remaining_items


class MealieProgressUI:
    """
    Rich terminal UI for Mealie operations with progress bars and status indicators.
    """

    def __init__(self, use_rich: bool = True):
        self.use_rich = use_rich and RICH_AVAILABLE
        self.console = Console() if self.use_rich else None
        self.current_operation: Optional[str] = None
        self.stats = ProgressStats(total=0)

    def show_welcome(self):
        """Display clean welcome screen."""
        print(MEALIE_HEADER)
        print("Initializing system...")

    def show_system_status(self, status: str = "ready"):
        """Show clean system status."""
        if status == "ready":
            print("System ready.")
        elif status == "working":
            print("Processing...")
        elif status == "complete":
            print("Operation complete.")
        else:
            print("Error: System issue detected.")
        print()

    def start_operation(self, operation: str, total_items: int, description: str = ""):
        """Start a new operation with progress tracking."""
        self.current_operation = operation
        self.stats = ProgressStats(total=total_items, start_time=time.time())

        welcome_msg = f"🚀 {operation}"
        if description:
            welcome_msg += f" - {description}"
        welcome_msg += f"\n📊 Processing {total_items} items"

        if self.use_rich:
            self.console.print(Panel(welcome_msg, title="Operation Started", border_style="blue"))
        else:
            print(welcome_msg)
            print("=" * 60)

    def update_progress(self, completed: int = None, failed: int = None, status_msg: str = ""):
        """Update progress with minimal output."""
        if completed is not None:
            self.stats.completed = completed
        if failed is not None:
            self.stats.failed = failed
        self.stats.last_update = time.time()

        # Only show progress at key milestones (every 25%, 50%, 75%, 100%)
        percent = (self.stats.completed + self.stats.failed) / self.stats.total * 100
        milestone = percent >= 25 and (self.stats.completed + self.stats.failed) % max(1, self.stats.total // 4) == 0

        if milestone or (self.stats.completed + self.stats.failed) == self.stats.total:
            if self.stats.completed + self.stats.failed == self.stats.total:
                print(f"Progress: {self.stats.completed}/{self.stats.total} complete")
            else:
                print(f"Progress: {int(percent)}% ({self.stats.completed}/{self.stats.total})")

    def _create_basic_progress_bar(self, percentage: float, width: int = 20) -> str:
        """Create a simple ASCII progress bar."""
        filled = int(width * percentage / 100)
        bar = "█" * filled + "░" * (width - filled)
        return f"[{bar}]"

    def show_status(self, message: str, style: str = "info"):
        """Show clean status message."""
        if style == "error":
            print(f"Error: {message}")
        elif style == "warning":
            print(f"Warning: {message}")
        else:
            print(message)

    def show_spinner(self, message: str, duration: float = None):
        """Show simple status for long operations."""
        print(f"{message}...")
        if duration:
            time.sleep(duration)
            print("Complete.")

    def complete_operation(self, final_status: str = "success"):
        """Complete operation with essential summary."""
        elapsed = self.stats.elapsed_time

        if final_status == "success":
            print(f"Completed: {self.stats.completed}/{self.stats.total} items in {elapsed:.1f}s")
        elif final_status == "partial":
            print(f"Partial: {self.stats.completed}/{self.stats.total} completed, {self.stats.failed} failed ({elapsed:.1f}s)")
        else:
            print(f"Failed: {self.stats.failed} errors in {elapsed:.1f}s")

    def create_timer(self, operation_name: str) -> 'Timer':
        """Create a timer for measuring operation duration."""
        return Timer(operation_name, self.use_rich, self.console)


class Timer:
    """Simple timer for measuring operation duration."""

    def __init__(self, operation: str, use_rich: bool = True, console: Optional[Console] = None):
        self.operation = operation
        self.use_rich = use_rich
        self.console = console
        self.start_time: Optional[float] = None
        self.end_time: Optional[float] = None

    def __enter__(self):
        self.start_time = time.time()
        if self.use_rich and self.console:
            self.console.print(f"⏱️  Started: {self.operation}", style="blue")
        else:
            print(f"⏱️  Started: {self.operation} at {datetime.now().strftime('%H:%M:%S')}")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end_time = time.time()
        duration = self.end_time - self.start_time

        if self.use_rich and self.console:
            style = "green" if exc_type is None else "red"
            self.console.print(f"⏱️  Completed: {self.operation} in {duration:.1f}s", style=style)
        else:
            status = "success" if exc_type is None else "failed"
            print(f"⏱️  Completed: {self.operation} in {duration:.1f}s ({status})")


# Global UI instance for easy access
ui = MealieProgressUI()


def create_progress_bar(iterable, description="Processing", total=None):
    """
    Create a progress bar for any iterable.

    Args:
        iterable: Items to iterate over
        description: Description for progress bar
        total: Total number of items (if known)

    Returns:
        Iterator that yields items with progress updates
    """
    if total is None:
        total = len(iterable) if hasattr(iterable, '__len__') else None

    if ui.use_rich and total:
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            TimeElapsedColumn(),
            TimeRemainingColumn(),
            console=ui.console
        ) as progress:
            task = progress.add_task(description, total=total)

            for item in iterable:
                yield item
                progress.update(task, advance=1)
    else:
        # Fallback to basic progress
        count = 0
        for item in iterable:
            if count % 10 == 0 and total:
                print(f"📊 {description}: {count}/{total}")
            count += 1
            yield item


# Convenience functions for common operations
def show_import_progress(current: int, total: int, operation: str = "Importing"):
    """Show import progress with ETA."""
    percent = current / total * 100
    ui.show_status(f"{operation}: {current}/{total} ({percent:.1f}%)", "info")


def show_parsing_status(recipe_name: str, status: str):
    """Show parsing status for individual recipes."""
    if status == "success":
        ui.show_status(f"✅ Parsed: {recipe_name}", "success")
    elif status == "failed":
        ui.show_status(f"❌ Failed: {recipe_name}", "error")
    else:
        ui.show_status(f"🔄 Processing: {recipe_name}", "info")


def show_batch_summary(completed: int, failed: int, total_time: float):
    """Show summary of batch operations."""
    success_rate = completed / (completed + failed) * 100 if (completed + failed) > 0 else 0
    ui.show_status(f"Batch complete: {completed} success, {failed} failed ({success_rate:.1f}% success) in {total_time:.1f}s", "info")


if __name__ == "__main__":
    # Demo the UI
    print("🎨 Mealie Progress UI Demo")
    print("=" * 40)

    # Test basic functionality
    ui.start_operation("Demo Operation", 100, "Testing progress UI")

    for i in range(0, 101, 10):
        ui.update_progress(completed=i, failed=0, status_msg=f"Processing item {i}")
        time.sleep(0.2)

    ui.complete_operation("success")

    # Test timer
    with ui.create_timer("Demo Timer"):
        time.sleep(1)

    print("\n✨ UI Demo Complete!")
