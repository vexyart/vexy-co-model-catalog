#!/usr/bin/env -S uv run -s
# /// script
# dependencies = ["rich", "pathlib", "ast"]
# ///
# this_file: scripts/code_quality_checker.py

"""
Comprehensive code quality checker for Python codebase.
Analyzes type hints coverage, docstring completeness, complexity, and code standards.
"""

import ast
import sys
from dataclasses import dataclass, field
from pathlib import Path
# Modern Python uses built-in generics instead of typing imports

from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress
from rich.table import Table

# Quality assessment thresholds
MIN_FUNCTIONS_FOR_ANALYSIS = 10
MIN_FILE_COUNT_FOR_COMPREHENSIVE_ANALYSIS = 50
MIN_MODULE_QUALITY_THRESHOLD = 6
MIN_DOCSTRING_WORDS = 2
EXCELLENT_COVERAGE_THRESHOLD = 90
GOOD_COVERAGE_THRESHOLD = 80
ACCEPTABLE_COVERAGE_THRESHOLD = 70
HIGH_QUALITY_RATING_THRESHOLD = 90
MEDIUM_QUALITY_RATING_THRESHOLD = 70
MIN_COMPLEXITY_ISSUES_FOR_WARNING = 5
MIN_ISSUES_FOR_DETAILED_ANALYSIS = 10
MAX_MCCABE_COMPLEXITY = 10
MAX_FUNCTION_LINES = 50
MAX_FUNCTION_PARAMETERS = 6


@dataclass
class QualityMetrics:
    """Container for code quality metrics."""

    total_functions: int = 0
    documented_functions: int = 0
    typed_functions: int = 0
    total_classes: int = 0
    documented_classes: int = 0
    typed_methods: int = 0
    total_methods: int = 0
    complexity_issues: list[str] = field(default_factory=list)
    missing_docstrings: list[str] = field(default_factory=list)
    missing_type_hints: list[str] = field(default_factory=list)
    code_smells: list[str] = field(default_factory=list)

    @property
    def docstring_coverage(self) -> float:
        """Calculate docstring coverage percentage."""
        total = self.total_functions + self.total_classes
        documented = self.documented_functions + self.documented_classes
        return (documented / total * 100) if total > 0 else 100.0

    @property
    def type_hint_coverage(self) -> float:
        """Calculate type hint coverage percentage."""
        total = self.total_functions + self.total_methods
        typed = self.typed_functions + self.typed_methods
        return (typed / total * 100) if total > 0 else 100.0

    @property
    def overall_quality_score(self) -> float:
        """Calculate overall quality score (0-100)."""
        docstring_weight = 0.4
        typing_weight = 0.4
        complexity_weight = 0.2

        docstring_score = self.docstring_coverage
        typing_score = self.type_hint_coverage

        # Deduct points for complexity issues and code smells
        complexity_deduction = min(len(self.complexity_issues) * 5, 30)
        smell_deduction = min(len(self.code_smells) * 3, 20)
        complexity_score = max(100 - complexity_deduction - smell_deduction, 0)

        return (
            docstring_score * docstring_weight +
            typing_score * typing_weight +
            complexity_score * complexity_weight
        )


class CodeQualityAnalyzer:
    """Analyzes Python code for quality metrics."""

    def __init__(self):
        self.console = Console()

    def analyze_file(self, file_path: Path) -> QualityMetrics:
        """Analyze a single Python file for quality metrics."""
        try:
            with open(file_path, encoding='utf-8') as f:
                content = f.read()

            tree = ast.parse(content)
            metrics = QualityMetrics()

            for node in ast.walk(tree):
                self._analyze_node(node, metrics, file_path)

            return metrics

        except Exception as e:
            self.console.print(f"[red]Error analyzing {file_path}: {e}[/red]")
            return QualityMetrics()

    def _analyze_node(self, node: ast.AST, metrics: QualityMetrics, file_path: Path) -> None:
        """Analyze a single AST node."""
        file_name = file_path.name

        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
            self._analyze_function(node, metrics, file_name)
        elif isinstance(node, ast.ClassDef):
            self._analyze_class(node, metrics, file_name)

    def _analyze_function(self, node: ast.FunctionDef, metrics: QualityMetrics, file_name: str) -> None:
        """Analyze function definition."""
        func_name = f"{file_name}:{node.name}()"

        # Count function
        if node.name.startswith('_') and not node.name.startswith('__'):
            # Private method
            metrics.total_methods += 1
            if self._has_return_type_hint(node):
                metrics.typed_methods += 1
        else:
            # Public function
            metrics.total_functions += 1
            if self._has_return_type_hint(node):
                metrics.typed_functions += 1

        # Check docstring
        if ast.get_docstring(node):
            if node.name.startswith('_') and not node.name.startswith('__'):
                pass  # Don't count private method docstrings for now
            else:
                metrics.documented_functions += 1
        # Skip dunder methods for docstring requirements
        elif not node.name.startswith('__'):
            metrics.missing_docstrings.append(func_name)

        # Check type hints
        if not self._has_comprehensive_type_hints(node):
            metrics.missing_type_hints.append(func_name)

        # Check complexity
        complexity = self._calculate_complexity(node)
        if complexity > MAX_MCCABE_COMPLEXITY:  # McCabe complexity threshold
            metrics.complexity_issues.append(f"{func_name} has complexity {complexity}")

        # Check for code smells
        self._check_code_smells(node, metrics, func_name)

    def _analyze_class(self, node: ast.ClassDef, metrics: QualityMetrics, file_name: str) -> None:
        """Analyze class definition."""
        class_name = f"{file_name}:{node.name}"

        metrics.total_classes += 1

        # Check docstring
        if ast.get_docstring(node):
            metrics.documented_classes += 1
        else:
            metrics.missing_docstrings.append(class_name)

        # Analyze class methods
        for item in node.body:
            if isinstance(item, ast.FunctionDef | ast.AsyncFunctionDef):
                self._analyze_function(item, metrics, file_name)

    def _has_return_type_hint(self, node: ast.FunctionDef) -> bool:
        """Check if function has return type hint."""
        return node.returns is not None

    def _has_comprehensive_type_hints(self, node: ast.FunctionDef) -> bool:
        """Check if function has comprehensive type hints."""
        # Skip dunder methods and some special cases
        if node.name in ['__init__', '__str__', '__repr__', '__eq__', '__hash__']:
            return True

        # Check arguments (skip 'self' and 'cls')
        start_idx = 1 if node.args.args and node.args.args[0].arg in ['self', 'cls'] else 0

        for arg in node.args.args[start_idx:]:
            if arg.annotation is None:
                return False

        # Check return type (skip __init__ methods)
        return not (node.name != '__init__' and node.returns is None)

    def _calculate_complexity(self, node: ast.FunctionDef) -> int:
        """Calculate cyclomatic complexity of a function."""
        complexity = 1  # Base complexity

        for child in ast.walk(node):
            if isinstance(child, ast.If | ast.While | ast.For | ast.AsyncFor | ast.ExceptHandler | (ast.And | ast.Or) | ast.comprehension):
                complexity += 1

        return complexity

    def _check_code_smells(self, node: ast.FunctionDef, metrics: QualityMetrics, func_name: str) -> None:
        """Check for common code smells."""
        # Long function (>50 lines)
        if hasattr(node, 'lineno') and hasattr(node, 'end_lineno'):
            if node.end_lineno and node.lineno:
                lines = node.end_lineno - node.lineno
                if lines > MAX_FUNCTION_LINES:
                    metrics.code_smells.append(f"{func_name} is too long ({lines} lines)")

        # Too many parameters
        if len(node.args.args) > MAX_FUNCTION_PARAMETERS:
            metrics.code_smells.append(f"{func_name} has too many parameters ({len(node.args.args)})")

        # Nested functions (potential complexity)
        function_defs = [child for child in ast.walk(node) if isinstance(child, ast.FunctionDef) and child != node]
        if len(function_defs) > 2:
            metrics.code_smells.append(f"{func_name} has too many nested functions ({len(function_defs)})")

    def analyze_directory(self, src_dir: Path) -> dict[str, QualityMetrics]:
        """Analyze all Python files in a directory."""
        python_files = list(src_dir.rglob("*.py"))
        results = {}

        with Progress() as progress:
            task = progress.add_task("[cyan]Analyzing files...", total=len(python_files))

            for file_path in python_files:
                # Skip __pycache__ and other generated files
                if "__pycache__" in str(file_path) or ".pyc" in str(file_path):
                    progress.advance(task)
                    continue

                rel_path = file_path.relative_to(src_dir)
                results[str(rel_path)] = self.analyze_file(file_path)
                progress.advance(task)

        return results

    def aggregate_metrics(self, file_metrics: dict[str, QualityMetrics]) -> QualityMetrics:
        """Aggregate metrics from multiple files."""
        total = QualityMetrics()

        for metrics in file_metrics.values():
            total.total_functions += metrics.total_functions
            total.documented_functions += metrics.documented_functions
            total.typed_functions += metrics.typed_functions
            total.total_classes += metrics.total_classes
            total.documented_classes += metrics.documented_classes
            total.typed_methods += metrics.typed_methods
            total.total_methods += metrics.total_methods
            total.complexity_issues.extend(metrics.complexity_issues)
            total.missing_docstrings.extend(metrics.missing_docstrings)
            total.missing_type_hints.extend(metrics.missing_type_hints)
            total.code_smells.extend(metrics.code_smells)

        return total

    def generate_report(self, file_metrics: dict[str, QualityMetrics]) -> None:
        """Generate comprehensive quality report."""
        total_metrics = self.aggregate_metrics(file_metrics)

        # Overall summary
        self._print_summary(total_metrics)

        # Detailed metrics by file
        self._print_file_details(file_metrics)

        # Issues and recommendations
        self._print_issues(total_metrics)

        # Quality recommendations
        self._print_recommendations(total_metrics)

    def _print_summary(self, metrics: QualityMetrics) -> None:
        """Print overall quality summary."""
        summary_table = Table(title="📊 Code Quality Summary", show_header=True)
        summary_table.add_column("Metric", style="cyan")
        summary_table.add_column("Value", style="green")
        summary_table.add_column("Coverage", style="yellow")
        summary_table.add_column("Status", justify="center")

        # Quality score
        score = metrics.overall_quality_score
        score_status = "🟢 Excellent" if score >= 90 else "🟡 Good" if score >= 80 else "🟠 Fair" if score >= 70 else "🔴 Needs Work"

        summary_table.add_row("Overall Quality Score", f"{score:.1f}/100", "", score_status)

        # Docstring coverage
        doc_coverage = metrics.docstring_coverage
        doc_status = "✅" if doc_coverage >= 90 else "⚠️" if doc_coverage >= 70 else "❌"
        summary_table.add_row(
            "Docstring Coverage",
            f"{metrics.documented_functions + metrics.documented_classes}/{metrics.total_functions + metrics.total_classes}",
            f"{doc_coverage:.1f}%",
            doc_status
        )

        # Type hint coverage
        type_coverage = metrics.type_hint_coverage
        type_status = "✅" if type_coverage >= 90 else "⚠️" if type_coverage >= 70 else "❌"
        summary_table.add_row(
            "Type Hint Coverage",
            f"{metrics.typed_functions + metrics.typed_methods}/{metrics.total_functions + metrics.total_methods}",
            f"{type_coverage:.1f}%",
            type_status
        )

        # Issues summary
        total_issues = len(metrics.complexity_issues) + len(metrics.code_smells)
        issue_status = "✅" if total_issues == 0 else "⚠️" if total_issues < 10 else "❌"
        summary_table.add_row("Code Issues", str(total_issues), "", issue_status)

        self.console.print(summary_table)

    def _print_file_details(self, file_metrics: dict[str, QualityMetrics]) -> None:
        """Print detailed metrics by file."""
        if not file_metrics:
            return

        file_table = Table(title="📁 File Quality Details", show_header=True)
        file_table.add_column("File", style="cyan")
        file_table.add_column("Functions", justify="right")
        file_table.add_column("Classes", justify="right")
        file_table.add_column("Doc Coverage", justify="right")
        file_table.add_column("Type Coverage", justify="right")
        file_table.add_column("Issues", justify="right")
        file_table.add_column("Quality", justify="center")

        # Sort files by quality score
        sorted_files = sorted(
            file_metrics.items(),
            key=lambda x: x[1].overall_quality_score,
            reverse=True
        )

        for file_path, metrics in sorted_files[:15]:  # Show top 15 files
            doc_coverage = metrics.docstring_coverage
            type_coverage = metrics.type_hint_coverage
            issues = len(metrics.complexity_issues) + len(metrics.code_smells)
            quality_score = metrics.overall_quality_score

            quality_icon = "🟢" if quality_score >= 90 else "🟡" if quality_score >= 80 else "🟠" if quality_score >= 70 else "🔴"

            file_table.add_row(
                file_path,
                str(metrics.total_functions),
                str(metrics.total_classes),
                f"{doc_coverage:.0f}%",
                f"{type_coverage:.0f}%",
                str(issues),
                f"{quality_icon} {quality_score:.0f}"
            )

        self.console.print(file_table)

    def _print_issues(self, metrics: QualityMetrics) -> None:
        """Print detected issues."""
        if metrics.missing_docstrings or metrics.missing_type_hints or metrics.complexity_issues:
            self.console.print("\\n[bold red]🔍 Detected Issues[/bold red]")

            # Missing docstrings
            if metrics.missing_docstrings:
                panel_content = "\\n".join(metrics.missing_docstrings[:10])
                if len(metrics.missing_docstrings) > 10:
                    panel_content += f"\\n... and {len(metrics.missing_docstrings) - 10} more"

                self.console.print(Panel(
                    panel_content,
                    title=f"📝 Missing Docstrings ({len(metrics.missing_docstrings)})",
                    border_style="yellow"
                ))

            # Missing type hints
            if metrics.missing_type_hints:
                panel_content = "\\n".join(metrics.missing_type_hints[:10])
                if len(metrics.missing_type_hints) > 10:
                    panel_content += f"\\n... and {len(metrics.missing_type_hints) - 10} more"

                self.console.print(Panel(
                    panel_content,
                    title=f"🏷️ Missing Type Hints ({len(metrics.missing_type_hints)})",
                    border_style="blue"
                ))

            # Complexity issues
            if metrics.complexity_issues:
                panel_content = "\\n".join(metrics.complexity_issues[:5])
                if len(metrics.complexity_issues) > 5:
                    panel_content += f"\\n... and {len(metrics.complexity_issues) - 5} more"

                self.console.print(Panel(
                    panel_content,
                    title=f"🔄 Complexity Issues ({len(metrics.complexity_issues)})",
                    border_style="red"
                ))

    def _print_recommendations(self, metrics: QualityMetrics) -> None:
        """Print quality improvement recommendations."""
        recommendations = []

        # Docstring recommendations
        if metrics.docstring_coverage < 90:
            missing_count = len(metrics.missing_docstrings)
            recommendations.append(f"📝 Add docstrings to {missing_count} functions/classes to improve documentation coverage")

        # Type hint recommendations
        if metrics.type_hint_coverage < 90:
            missing_count = len(metrics.missing_type_hints)
            recommendations.append(f"🏷️ Add type hints to {missing_count} functions to improve type safety")

        # Complexity recommendations
        if metrics.complexity_issues:
            recommendations.append(f"🔄 Refactor {len(metrics.complexity_issues)} complex functions to reduce cyclomatic complexity")

        # Code smell recommendations
        if metrics.code_smells:
            recommendations.append(f"🧽 Address {len(metrics.code_smells)} code smells to improve maintainability")

        if recommendations:
            self.console.print("\\n[bold cyan]💡 Improvement Recommendations[/bold cyan]")
            for i, rec in enumerate(recommendations, 1):
                self.console.print(f"{i}. {rec}")

        # Quality milestones
        score = metrics.overall_quality_score
        if score < 70:
            self.console.print("\\n[red]🎯 Target: Reach 70+ quality score by addressing critical issues[/red]")
        elif score < 80:
            self.console.print("\\n[yellow]🎯 Target: Reach 80+ quality score by improving type coverage[/yellow]")
        elif score < 90:
            self.console.print("\\n[blue]🎯 Target: Reach 90+ quality score by completing documentation[/blue]")
        else:
            self.console.print("\\n[green]✨ Excellent! Your code meets production quality standards![/green]")


def main():
    """Main entry point for code quality analysis."""
    console = Console()

    # Find source directory
    script_dir = Path(__file__).parent
    src_dir = script_dir.parent / "src" / "vexy_co_model_catalog"

    if not src_dir.exists():
        console.print(f"[red]Source directory not found: {src_dir}[/red]")
        sys.exit(1)

    console.print(f"[cyan]Analyzing code quality in: {src_dir}[/cyan]\\n")

    analyzer = CodeQualityAnalyzer()
    file_metrics = analyzer.analyze_directory(src_dir)

    if not file_metrics:
        console.print("[yellow]No Python files found to analyze[/yellow]")
        sys.exit(1)

    analyzer.generate_report(file_metrics)

    # Return quality score as exit code (for CI/CD)
    total_metrics = analyzer.aggregate_metrics(file_metrics)
    quality_score = total_metrics.overall_quality_score

    if quality_score >= 80:
        sys.exit(0)  # Success
    elif quality_score >= 60:
        sys.exit(1)  # Warning
    else:
        sys.exit(2)  # Critical


if __name__ == "__main__":
    main()
