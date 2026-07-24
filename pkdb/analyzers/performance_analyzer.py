import re
from typing import Dict, List


class ThreadDivergenceAnalyzer:
    def __init__(self, gdb_controller):
        self.gdb = gdb_controller
        self.warp_size = 32

    def analyze_current_state(self) -> Dict:
        threads = self._get_cuda_thread_states()
        if not threads:
            return {"error": "No CUDA thread information available. Are you at a kernel breakpoint?"}

        warps = self._group_by_warp(threads)
        divergent_warps = []

        for warp_id, warp_threads in warps.items():
            pcs = [t["pc"] for t in warp_threads if "pc" in t]
            unique_pcs = set(pcs)

            if len(unique_pcs) > 1:
                # Calculate divergence as % of threads NOT on the dominant path
                from collections import Counter

                pc_counts = Counter(pcs)
                max_count = max(pc_counts.values())
                diverging_threads = len(pcs) - max_count
                divergence_pct = (diverging_threads / len(pcs)) * 100

                divergent_warps.append(
                    {
                        "warp_id": warp_id,
                        "divergence_pct": divergence_pct,
                        "thread_count": len(warp_threads),
                        "unique_paths": len(unique_pcs),
                        "diverging_threads": diverging_threads,
                    }
                )

        total_warps = len(warps)
        divergent_count = len(divergent_warps)

        return {
            "total_warps": total_warps,
            "divergent_warps": divergent_count,
            "divergence_ratio": divergent_count / total_warps if total_warps > 0 else 0,
            "details": divergent_warps,
            "total_threads": len(threads),
        }

    def _get_cuda_thread_states(self) -> List[Dict]:
        responses = self.gdb._send_mi_command('interpreter-exec console "info cuda threads"')
        threads = []

        for resp in responses:
            if resp.get("type") == "console":
                output = resp.get("payload", "")
                for line in output.split("\\n"):
                    # Parse range format: (0,0,0) (0,0,0) (1,0,0) (0,127,0) 256 0x00007fff...
                    range_match = re.search(
                        r"\*?\s*\((\d+),(\d+),(\d+)\)\s+\((\d+),(\d+),(\d+)\)\s+\((\d+),(\d+),(\d+)\)\s+\((\d+),(\d+),(\d+)\)\s+(\d+)\s+(0x[0-9a-f]+)",
                        line,
                    )
                    if range_match:
                        # Extract block range and thread range
                        from_block = (int(range_match.group(1)), int(range_match.group(2)), int(range_match.group(3)))
                        from_thread = (int(range_match.group(4)), int(range_match.group(5)), int(range_match.group(6)))
                        to_thread = (int(range_match.group(10)), int(range_match.group(11)), int(range_match.group(12)))
                        pc = range_match.group(14)

                        # Generate individual thread entries for all threads in range
                        for tid_x in range(from_thread[0], to_thread[0] + 1):
                            threads.append(
                                {"block": from_block[0], "thread": tid_x, "pc": pc, "warp": tid_x // self.warp_size}
                            )

        return threads

    def _group_by_warp(self, threads: List[Dict]) -> Dict[int, List[Dict]]:
        warps = {}
        for t in threads:
            warp_key = (t["block"], t["warp"])
            if warp_key not in warps:
                warps[warp_key] = []
            warps[warp_key].append(t)
        return warps

    def format_report(self, analysis: Dict) -> str:
        if "error" in analysis:
            return f"Error: {analysis['error']}"

        lines = ["\n=== Thread Divergence Analysis ==="]
        lines.append(f"Total Threads: {analysis.get('total_threads', 'N/A')}")
        lines.append(f"Total Warps: {analysis['total_warps']}")
        lines.append(f"Divergent Warps: {analysis['divergent_warps']}")
        lines.append(f"Divergence Ratio: {analysis['divergence_ratio']*100:.1f}%")

        if analysis["details"]:
            lines.append("\nDivergent Warps:")
            for detail in analysis["details"][:5]:
                warp_id = detail["warp_id"]
                if isinstance(warp_id, tuple):
                    warp_id = f"Block {warp_id[0]}, Warp {warp_id[1]}"
                lines.append(
                    f"  {warp_id}: {detail['unique_paths']} paths, "
                    f"{detail['diverging_threads']}/{detail['thread_count']} threads diverging "
                    f"({detail['divergence_pct']:.1f}%)"
                )

        if analysis["divergence_ratio"] > 0.5:
            lines.append("\n!!! High divergence: >50% of warps have divergent threads")
            lines.append("💡 Consider reordering branches or using warp primitives")
        elif analysis["divergent_warps"] > 0:
            lines.append(f"\n! Moderate divergence: {analysis['divergence_ratio']*100:.0f}% of warps affected")
            lines.append("💡 Profile to determine if this impacts performance")
        else:
            lines.append("\nNo divergence detected - all warps executing same path")

        return "\n".join(lines)


class PerformanceAnalyzer:
    def __init__(self, gdb_controller):
        self.divergence = ThreadDivergenceAnalyzer(gdb_controller)
