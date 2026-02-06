#!/usr/bin/env python3
"""
Parallel processing utilities for bulk operations.
Handles recipe processing with proper error handling.
"""

import sys
import os
# Add parent directory to path for imports from project root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import concurrent.futures
from typing import List, Callable, Any
import time
from tools.progress_ui import ui


class ParallelProcessor:
    """
    Generic parallel processor for recipe operations.
    """

    def __init__(self, max_workers: int = 20):
        self.max_workers = max_workers

    def process_batch(
        self,
        items: List[Any],
        process_func: Callable[[Any], Any],
        description: str = "Processing"
    ) -> List[Any]:
        """
        Process items in parallel with progress reporting.

        Args:
            items: List of items to process
            process_func: Function to process each item
            description: Progress description

        Returns:
            List of results (in original order)
        """
        results = [None] * len(items)

        ui.start_operation(description, len(items), f"Using {self.max_workers} workers")
        start_time = time.time()

        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all tasks
            future_to_index = {
                executor.submit(process_func, item): i
                for i, item in enumerate(items)
            }

            # Collect results as they complete
            completed = 0
            failed = 0
            for future in concurrent.futures.as_completed(future_to_index):
                index = future_to_index[future]
                try:
                    result = future.result()
                    results[index] = result
                except Exception as e:
                    ui.show_status(f"Error processing item {index}: {e}", "error")
                    results[index] = None
                    failed += 1

                completed += 1
                if completed % 10 == 0 or completed == len(items):
                    ui.update_progress(completed=completed, failed=failed,
                                     status_msg=f"{completed}/{len(items)} processed")

        elapsed = time.time() - start_time
        success_count = sum(1 for r in results if r is not None)

        if len(items) > 0:
            success_rate = success_count / len(items) * 100
            print(f"Completed: {success_count}/{len(items)} items in {elapsed:.1f}s")
        else:
            print("Completed: 0 items")
        return results
