import sys
import argparse
sys.path.append('.')

from src.data_generator import FinanceDataGenerator
from src.config import Config
from src.utils import save_dataframe, print_section

def main():
    parser = argparse.ArgumentParser(description='Generate synthetic financial data')
    parser.add_argument('--ar-records', type=int, default=100, help='Number of AR records')
    parser.add_argument('--claims-records', type=int, default=200, help='Number of expense claims')
    parser.add_argument('--budget-years', type=int, default=2, help='Number of budget years')
    parser.add_argument('--output-dir', type=str, default='data/sample', help='Output directory')
    
    args = parser.parse_args()
    
    Config.ensure_directories()
    
    print_section("FINANCE DATA GENERATOR")
    
    print("Initializing data generator...")
    generator = FinanceDataGenerator()
    
    print(f"\nGenerating {args.ar_records} Accounts Receivable records...")
    ar_df = generator.generate_accounts_receivable(n=args.ar_records)
    save_dataframe(ar_df, 'accounts_receivable.xlsx', args.output_dir)
    
    print("\nGenerating payment records...")
    payments_df = generator.generate_payments(ar_df)
    save_dataframe(payments_df, 'payments.xlsx', args.output_dir)
    
    print("\nGenerating general ledger entries...")
    gl_df = generator.generate_general_ledger(ar_df)
    save_dataframe(gl_df, 'general_ledger.xlsx', args.output_dir)
    
    print(f"\nGenerating {args.budget_years} years of budget data...")
    budget_df = generator.generate_budget_forecast(n_years=args.budget_years)
    save_dataframe(budget_df, 'budget_forecast.xlsx', args.output_dir)
    
    print(f"\nGenerating {args.claims_records} expense claims...")
    claims_df = generator.generate_expense_claims(n=args.claims_records)
    save_dataframe(claims_df, 'expense_claims.xlsx', args.output_dir)
    
    print_section("DATA GENERATION COMPLETE")
    print(f"All files saved to: {args.output_dir}/")
    print("\nSummary:")
    print(f"  - Accounts Receivable: {len(ar_df)} records")
    print(f"  - Payments: {len(payments_df)} records")
    print(f"  - General Ledger: {len(gl_df)} records")
    print(f"  - Budget Forecast: {len(budget_df)} records")
    print(f"  - Expense Claims: {len(claims_df)} records")

if __name__ == "__main__":
    main()