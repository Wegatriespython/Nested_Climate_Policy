import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

def load_and_prepare_data():
    """Load and prepare the sensitivity analysis results."""
    df = pd.read_csv('sensitivity_results.csv')
    # Calculate technology change
    df['Δη'] = df['η_1'] - df['η_0']
    # Calculate productivity change
    df['ΔA'] = df['A_1'] - df['A_0']
    return df

def create_success_rate_plots(df):
    """Create various success rate visualizations."""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Success by A_0
    success_by_A = df.groupby('A_0')['success'].mean()
    axes[0,0].plot(success_by_A.index, success_by_A.values)
    axes[0,0].set_title('Success Rate by Initial Productivity (A₀)')
    axes[0,0].set_xlabel('A₀')
    axes[0,0].set_ylabel('Success Rate')
    axes[0,0].grid(True)

    # Success by Δη
    success_by_deta = df.groupby('Δη')['success'].mean()
    axes[0,1].plot(success_by_deta.index, success_by_deta.values)
    axes[0,1].set_title('Success Rate by Technology Change (Δη)')
    axes[0,1].set_xlabel('Δη')
    axes[0,1].set_ylabel('Success Rate')
    axes[0,1].grid(True)

    # Success by skill factor
    success_by_sf = df.groupby('skill_factor')['success'].mean()
    axes[1,0].plot(success_by_sf.index, success_by_sf.values)
    axes[1,0].set_title('Success Rate by Skill Factor')
    axes[1,0].set_xlabel('Skill Factor')
    axes[1,0].set_ylabel('Success Rate')
    axes[1,0].grid(True)

    # Heatmap of A_0 vs Δη
    pivot_data = df.pivot_table(
        values='success',
        index='A_0',
        columns='Δη',
        aggfunc='mean'
    )
    im = axes[1,1].imshow(pivot_data, aspect='auto', cmap='viridis')
    axes[1,1].set_title('Success Rate Heatmap (A₀ vs Δη)')
    plt.colorbar(im, ax=axes[1,1])
    
    plt.tight_layout()
    plt.savefig('success_rate_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()

def analyze_labor_efficiency(df):
    """Analyze labor efficiency patterns in successful cases."""
    successful_cases = df[df['success']]
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Labor Efficiency vs Technology Change
    axes[0,0].scatter(successful_cases['Δη'], successful_cases['labor_efficiency'], alpha=0.5)
    axes[0,0].set_title('Labor Efficiency vs Technology Change')
    axes[0,0].set_xlabel('Δη')
    axes[0,0].set_ylabel('Labor Efficiency')
    axes[0,0].grid(True)

    # Box plot of Labor Efficiency by Skill Factor
    axes[0,1].boxplot([successful_cases[successful_cases['skill_factor'] == sf]['labor_efficiency'] 
                      for sf in successful_cases['skill_factor'].unique()])
    axes[0,1].set_title('Labor Efficiency Distribution by Skill Factor')
    axes[0,1].set_xlabel('Skill Factor')
    axes[0,1].set_ylabel('Labor Efficiency')
    axes[0,1].grid(True)

    # Heatmap of Labor Efficiency
    pivot_data = successful_cases.pivot_table(
        values='labor_efficiency',
        index='A_0',
        columns='skill_factor',
        aggfunc='mean'
    )
    im = axes[1,0].imshow(pivot_data, aspect='auto', cmap='viridis')
    axes[1,0].set_title('Average Labor Efficiency (A₀ vs Skill Factor)')
    plt.colorbar(im, ax=axes[1,0])

    # Labor Efficiency vs Output
    scatter = axes[1,1].scatter(successful_cases['Y_0'], successful_cases['labor_efficiency'], 
                              c=successful_cases['skill_factor'], cmap='viridis')
    axes[1,1].set_title('Labor Efficiency vs Output')
    axes[1,1].set_xlabel('Output (Y₀)')
    axes[1,1].set_ylabel('Labor Efficiency')
    plt.colorbar(scatter, ax=axes[1,1], label='Skill Factor')
    axes[1,1].grid(True)
    
    plt.tight_layout()
    plt.savefig('labor_efficiency_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()

def generate_summary_statistics(df):
    """Generate and save summary statistics."""
    # Open file with UTF-8 encoding
    with open('sensitivity_analysis_summary.txt', 'w', encoding='utf-8') as f:
        f.write("Sensitivity Analysis Summary\n")
        f.write("===========================\n\n")
        
        # Overall success rate
        f.write(f"Overall Success Rate: {df['success'].mean():.2%}\n\n")
        
        # Success rates by parameter ranges
        f.write("Success Rates by Initial Productivity (A0):\n")
        success_by_A = df.groupby('A_0')['success'].mean()
        for A, rate in success_by_A.items():
            f.write(f"  A0 = {A:.2f}: {rate:.2%}\n")
        f.write("\n")
        
        # Labor efficiency statistics
        successful_cases = df[df['success']]
        f.write("Labor Efficiency Statistics (Successful Cases):\n")
        f.write(f"  Mean: {successful_cases['labor_efficiency'].mean():.3f}\n")
        f.write(f"  Std Dev: {successful_cases['labor_efficiency'].std():.3f}\n")
        f.write(f"  Min: {successful_cases['labor_efficiency'].min():.3f}\n")
        f.write(f"  Max: {successful_cases['labor_efficiency'].max():.3f}\n\n")
        
        # Output statistics
        f.write("Output Statistics (Successful Cases):\n")
        f.write(f"  Mean: {successful_cases['Y_0'].mean():.3f}\n")
        f.write(f"  Std Dev: {successful_cases['Y_0'].std():.3f}\n")
        f.write(f"  Min: {successful_cases['Y_0'].min():.3f}\n")
        f.write(f"  Max: {successful_cases['Y_0'].max():.3f}\n")

def main():
    # Load data
    try:
        df = load_and_prepare_data()
        
        # Generate all analyses
        create_success_rate_plots(df)
        analyze_labor_efficiency(df)
        generate_summary_statistics(df)
        
        print("Analysis complete. Check the generated files for results.")
    except Exception as e:
        print(f"Error during analysis: {str(e)}")
        raise

if __name__ == "__main__":
    main()