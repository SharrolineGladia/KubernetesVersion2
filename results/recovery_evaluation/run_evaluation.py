"""
Quick Start Guide for Recovery Evaluation
==========================================

This script demonstrates how to run the recovery evaluation system.
Choose between simulated (fast, no dependencies) or live (real Kubernetes) evaluation.
"""

import os
import sys

def print_header():
    print("="*70)
    print("🚀 Kubernetes Anomaly Detection Framework - Recovery Evaluator")
    print("="*70)
    print()

def print_menu():
    print("Select evaluation mode:")
    print()
    print("1. Simulated Evaluation (RECOMMENDED)")
    print("   ✅ No Kubernetes cluster required")
    print("   ✅ Fast execution (~15 seconds for 25 injections)")
    print("   ✅ Reproducible results")
    print("   ✅ Realistic timing models")
    print()
    print("2. Live Evaluation (ADVANCED)")
    print("   ⚠️  Requires running Kubernetes cluster")
    print("   ⚠️  Requires Prometheus server")
    print("   ⚠️  Requires notification service")
    print("   ⏱️  Slower (5-10 minutes for 20 injections)")
    print("   ✅ Real-world results")
    print()
    print("3. View latest results")
    print()
    print("4. Exit")
    print()

def run_simulated():
    print("\n🔄 Running simulated evaluation...")
    print("="*70)
    
    # Import and run
    try:
        from simulated_evaluator import SimulatedRecoveryEvaluator
        
        # Initialize
        evaluator = SimulatedRecoveryEvaluator(seed=42)
        
        # Run evaluation
        print("\n⏳ Running 25 simulated injections...")
        df = evaluator.run_evaluation(num_injections=25)
        
        # Save results
        print("\n💾 Saving results...")
        evaluator.save_results()
        
        # Generate visualizations
        print("\n📊 Generating visualizations...")
        evaluator.generate_visualizations()
        
        # Print summary
        evaluator.print_summary_report()
        
        print("\n✅ Evaluation complete!")
        print(f"📁 Results saved in: {evaluator.results_dir}")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        print("\nTroubleshooting:")
        print("- Ensure you have installed dependencies: pip install -r requirements.txt")
        print("- Check that numpy, pandas, matplotlib are installed")

def run_live():
    print("\n⚠️  Live evaluation requires:")
    print("  1. Kubernetes cluster running")
    print("  2. Prometheus at http://localhost:9090")
    print("  3. Notification service at http://localhost:8003")
    print()
    
    response = input("Are all services running? (y/n): ")
    
    if response.lower() != 'y':
        print("\n⏸️  Aborted. Start services first.")
        return
    
    print("\n🔄 Running live evaluation...")
    print("="*70)
    
    try:
        # Check if model exists
        model_path = '../../ml_detector/models/anomaly_detector_scaleinvariant.pkl'
        if not os.path.exists(model_path):
            print(f"\n❌ Model not found: {model_path}")
            print("Train the model first using ml_detector/scripts/train_scaleinvariant_model.py")
            return
        
        from comprehensive_evaluator import ComprehensiveEvaluator
        
        # Initialize
        evaluator = ComprehensiveEvaluator(
            model_path=model_path,
            service_url='http://localhost:8003',
            prometheus_url='http://localhost:9090'
        )
        
        # Run evaluation
        print("\n⏳ Running 20 live injections (this will take 10-15 minutes)...")
        df = evaluator.run_comprehensive_evaluation(num_injections=20)
        
        # Generate visualizations
        print("\n📊 Generating visualizations...")
        evaluator.generate_visualizations()
        
        # Print summary
        evaluator.print_summary_report()
        
        print("\n✅ Evaluation complete!")
        print(f"📁 Results saved in: {evaluator.results_dir}")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        print("\nTroubleshooting:")
        print("- Check Kubernetes cluster: kubectl get pods")
        print("- Check Prometheus: curl http://localhost:9090/-/healthy")
        print("- Check notification service: curl http://localhost:8003/health")

def view_results():
    print("\n📊 Latest Results")
    print("="*70)
    
    # Find latest files
    import glob
    
    csv_files = glob.glob("simulated_evaluation_*.csv")
    json_files = glob.glob("simulated_metrics_*.json")
    plot_files = glob.glob("evaluation_plots_*.png")
    
    if not csv_files:
        print("\n❌ No results found. Run an evaluation first!")
        return
    
    # Get latest
    latest_csv = sorted(csv_files)[-1]
    latest_json = sorted(json_files)[-1] if json_files else None
    latest_plot = sorted(plot_files)[-1] if plot_files else None
    
    print(f"\n📄 Latest CSV:  {latest_csv}")
    if latest_json:
        print(f"📄 Latest JSON: {latest_json}")
    if latest_plot:
        print(f"📄 Latest Plot: {latest_plot}")
    
    # Read and display summary
    if latest_json:
        import json
        print("\n📊 Summary Metrics:")
        print("-"*70)
        
        with open(latest_json, 'r') as f:
            metrics = json.load(f)
        
        print(f"  Total Injections:       {metrics['total_injections']}")
        print(f"  Detection Success:      {metrics['detection_success_rate']:.1f}%")
        print(f"  Recovery Success:       {metrics['recovery_success_rate']:.1f}%")
        print(f"  Mean MTTR:              {metrics['mean_mttr']:.2f}s")
        print(f"  Mean E2E Latency:       {metrics['mean_e2e_latency']:.2f}s")
        print(f"  Anomaly Prob Reduction: {metrics['mean_anomaly_prob_reduction']:.3f}")
        print(f"  Recurrence Rate:        {metrics['recurrence_rate']:.1f}%")
    
    print("\n💡 To view visualizations, open:")
    if latest_plot:
        print(f"   {os.path.abspath(latest_plot)}")

def main():
    print_header()
    
    while True:
        print_menu()
        choice = input("Enter choice (1-4): ")
        
        if choice == '1':
            run_simulated()
            input("\nPress Enter to continue...")
        elif choice == '2':
            run_live()
            input("\nPress Enter to continue...")
        elif choice == '3':
            view_results()
            input("\nPress Enter to continue...")
        elif choice == '4':
            print("\n👋 Goodbye!")
            break
        else:
            print("\n❌ Invalid choice. Please enter 1-4.")
        
        print("\n")

if __name__ == "__main__":
    main()
