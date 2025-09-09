#!/usr/bin/env bash
# this_file: scripts/run_performance_gates.sh

# Performance Quality Gates Script
# Automated performance regression prevention for CI/CD pipelines

set -euo pipefail

# Script configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
BENCHMARK_SCRIPT="$SCRIPT_DIR/performance_benchmark.py"
RESULTS_DIR="$PROJECT_ROOT/performance_results"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

# Performance thresholds (can be overridden by environment variables)
CACHE_ACCESS_MAX_MS=${CACHE_ACCESS_MAX_MS:-100}
CLI_STARTUP_MAX_MS=${CLI_STARTUP_MAX_MS:-2000}
MEMORY_GROWTH_MAX_MB=${MEMORY_GROWTH_MAX_MB:-50}
MIN_CACHE_HIT_RATE=${MIN_CACHE_HIT_RATE:-75}

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

print_header() {
    echo -e "${CYAN}=====================================${NC}"
    echo -e "${CYAN} Performance Regression Gates v1.0  ${NC}"
    echo -e "${CYAN}=====================================${NC}"
    echo ""
}

print_success() {
    echo -e "${GREEN}✅ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

print_error() {
    echo -e "${RED}❌ $1${NC}"
}

print_info() {
    echo -e "${CYAN}ℹ️  $1${NC}"
}

setup_environment() {
    print_info "Setting up benchmark environment..."
    
    # Create results directory
    mkdir -p "$RESULTS_DIR"
    
    # Verify Python environment
    if ! command -v python3 &> /dev/null; then
        print_error "Python 3 is required but not found"
        exit 1
    fi
    
    # Verify uv is available for script dependencies
    if ! command -v uv &> /dev/null; then
        print_warning "uv not found, trying with system python"
        PYTHON_CMD="python3"
    else
        PYTHON_CMD="uv run"
        print_success "uv found, using optimized dependency management"
    fi
    
    export PYTHON_CMD
}

run_performance_benchmark() {
    print_info "Running comprehensive performance benchmark..."
    
    local result_file="$RESULTS_DIR/benchmark_$TIMESTAMP.log"
    
    # Set performance thresholds as environment variables
    export CACHE_ACCESS_MAX_MS
    export CLI_STARTUP_MAX_MS  
    export MEMORY_GROWTH_MAX_MB
    export MIN_CACHE_HIT_RATE
    
    # Run the benchmark with output capturing
    if $PYTHON_CMD "$BENCHMARK_SCRIPT" 2>&1 | tee "$result_file"; then
        local exit_code=${PIPESTATUS[0]}
        
        if [[ $exit_code -eq 0 ]]; then
            print_success "Performance benchmark completed successfully"
            return 0
        elif [[ $exit_code -eq 1 ]]; then
            print_error "CRITICAL: Performance regression detected!"
            print_error "Check results in: $result_file"
            return 1
        else
            print_error "Benchmark execution failed with code $exit_code"
            return 1
        fi
    else
        print_error "Failed to execute performance benchmark"
        return 1
    fi
}

analyze_trends() {
    print_info "Analyzing performance trends..."
    
    # Simple trend analysis of recent results
    local recent_results=($(ls -t "$RESULTS_DIR"/benchmark_*.log 2>/dev/null | head -5))
    
    if [[ ${#recent_results[@]} -lt 2 ]]; then
        print_info "Insufficient historical data for trend analysis"
        return 0
    fi
    
    print_info "Found ${#recent_results[@]} recent benchmark results"
    print_info "Latest: $(basename "${recent_results[0]}")"
    
    # Check for critical failures in recent runs
    local critical_count=0
    for result_file in "${recent_results[@]}"; do
        if grep -q "CRITICAL" "$result_file"; then
            ((critical_count++))
        fi
    done
    
    if [[ $critical_count -gt 1 ]]; then
        print_warning "Multiple critical performance issues detected recently"
        print_warning "Consider investigating performance regression patterns"
    fi
    
    return 0
}

cleanup_old_results() {
    print_info "Cleaning up old benchmark results..."
    
    # Keep only last 30 results to prevent disk usage growth
    local results_to_delete=($(ls -t "$RESULTS_DIR"/benchmark_*.log 2>/dev/null | tail -n +31))
    
    if [[ ${#results_to_delete[@]} -gt 0 ]]; then
        for old_result in "${results_to_delete[@]}"; do
            rm -f "$old_result"
        done
        print_info "Cleaned up ${#results_to_delete[@]} old benchmark results"
    fi
}

generate_summary_report() {
    print_info "Generating performance summary report..."
    
    local summary_file="$RESULTS_DIR/performance_summary.md"
    local latest_result=($(ls -t "$RESULTS_DIR"/benchmark_*.log 2>/dev/null | head -1))
    
    if [[ ${#latest_result[@]} -eq 0 ]]; then
        print_warning "No benchmark results found for summary"
        return 0
    fi
    
    cat > "$summary_file" << EOF
# Performance Regression Prevention Report

**Generated:** $(date)
**Latest Benchmark:** $(basename "${latest_result[0]}")

## Performance Thresholds
- Cache Access: < ${CACHE_ACCESS_MAX_MS}ms  
- CLI Startup: < ${CLI_STARTUP_MAX_MS}ms
- Memory Growth: < ${MEMORY_GROWTH_MAX_MB}MB
- Cache Hit Rate: > ${MIN_CACHE_HIT_RATE}%

## Latest Results
$(tail -20 "${latest_result[0]}" | sed 's/^/    /')

## Trend Analysis
$(ls -la "$RESULTS_DIR"/benchmark_*.log 2>/dev/null | tail -5 | awk '{print "- " $9 " (" $6 " " $7 " " $8 ")"}')

---
*Automated Performance Regression Prevention System v1.0*
EOF

    print_success "Summary report generated: $summary_file"
}

main() {
    print_header
    
    # Validate script execution environment  
    if [[ ! -f "$BENCHMARK_SCRIPT" ]]; then
        print_error "Benchmark script not found: $BENCHMARK_SCRIPT"
        exit 1
    fi
    
    # Execute benchmark pipeline
    setup_environment
    
    if run_performance_benchmark; then
        benchmark_result=0
        print_success "Performance quality gates: PASSED"
    else
        benchmark_result=1
        print_error "Performance quality gates: FAILED"
    fi
    
    # Post-benchmark analysis (always run for insights)
    analyze_trends
    cleanup_old_results  
    generate_summary_report
    
    # Final status
    echo ""
    if [[ $benchmark_result -eq 0 ]]; then
        print_success "✅ All performance quality gates passed!"
        print_info "System performance is within acceptable thresholds"
    else
        print_error "❌ Performance regression detected!"
        print_error "Review benchmark results and address performance issues"
        echo ""
        print_info "Common causes of performance regression:"
        print_info "  • Cache configuration changes"
        print_info "  • Memory leaks in new code"  
        print_info "  • Inefficient algorithms or data structures"
        print_info "  • Dependency version changes"
        echo ""
    fi
    
    exit $benchmark_result
}

# Execute main function if script is run directly
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    main "$@"
fi