#!/usr/bin/env python3
"""
Advanced EDA for IPL Score Prediction
Run this script to perform comprehensive exploratory data analysis
Uses the same data loading and cleaning functions as the training pipeline
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from sklearn.preprocessing import LabelEncoder
import sys
import os

# Add current directory to path to import utils
sys.path.append(os.path.dirname(__file__))
from utils import clean_data

warnings.filterwarnings('ignore')

# Set style
plt.style.use('default')
sns.set_palette("husl")
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)

def main():
    print("="*60)
    print("ADVANCED EDA FOR IPL SCORE PREDICTION")
    print("="*60)
    
    # 1. Load data using the same method as training pipeline
    print("\n1. LOADING AND CLEANING DATA...")
    try:
        # Use the same path as training pipeline
        df = pd.read_csv('content/ipl_data.csv')
        print(f"Original dataset shape: {df.shape}")
        
        # Apply the same cleaning as training pipeline
        df_cleaned = clean_data(df)
        print(f"Cleaned dataset shape: {df_cleaned.shape}")
        print("First few rows after cleaning:")
        print(df_cleaned.head())
        
        # Use cleaned data for analysis
        df = df_cleaned
        
    except Exception as e:
        print(f"Error: Could not load or clean ipl_data.csv: {e}")
        return
    
    # 2. Data quality analysis
    print("\n" + "="*50)
    print("2. DATA QUALITY ANALYSIS")
    print("="*50)
    
    print(f"Missing values:\n{df.isnull().sum()}")
    print(f"\nDuplicate rows: {df.duplicated().sum()}")
    print(f"\nData types:\n{df.dtypes}")
    
    # 3. Statistical summary
    print("\nStatistical Summary:")
    print(df.describe())
    
    # 4. Team performance analysis
    print("\n" + "="*50)
    print("3. TEAM PERFORMANCE ANALYSIS")
    print("="*50)
    
    # Batting team performance
    batting_stats = df.groupby('bat_team').agg({
        'total': ['mean', 'std', 'count'],
        'runs': 'mean',
        'wickets': 'mean'
    }).round(2)
    batting_stats.columns = ['avg_total', 'std_total', 'matches', 'avg_runs', 'avg_wickets']
    batting_stats = batting_stats.sort_values('avg_total', ascending=False)
    
    print("Top 5 Batting Teams:")
    print(batting_stats.head())
    
    # Bowling team performance
    bowling_stats = df.groupby('bowl_team').agg({
        'total': ['mean', 'std'],
        'wickets': 'mean'
    }).round(2)
    bowling_stats.columns = ['avg_total_conceded', 'std_total_conceded', 'avg_wickets_taken']
    bowling_stats = bowling_stats.sort_values('avg_total_conceded')
    
    print("\nTop 5 Bowling Teams (lowest conceded):")
    print(bowling_stats.head())
    
    # 5. Venue analysis
    print("\n" + "="*50)
    print("4. VENUE ANALYSIS")
    print("="*50)
    
    venue_stats = df.groupby('venue').agg({
        'total': ['mean', 'std', 'count'],
        'runs': 'mean',
        'wickets': 'mean'
    }).round(2)
    venue_stats.columns = ['avg_total', 'std_total', 'matches', 'avg_runs', 'avg_wickets']
    venue_stats = venue_stats.sort_values('avg_total', ascending=False)
    
    print("Top 10 High-Scoring Venues:")
    print(venue_stats.head(10))
    
    # 6. Player analysis
    print("\n" + "="*50)
    print("5. PLAYER ANALYSIS")
    print("="*50)
    
    # Top batsmen
    batsman_stats = df.groupby('batsman').agg({
        'runs': ['sum', 'mean', 'count'],
        'total': 'mean'
    }).round(2)
    batsman_stats.columns = ['total_runs', 'avg_runs', 'innings', 'avg_team_total']
    batsman_stats = batsman_stats.sort_values('total_runs', ascending=False)
    
    print("Top 10 Batsmen by Total Runs:")
    print(batsman_stats.head(10))
    
    # Top bowlers
    bowler_stats = df.groupby('bowler').agg({
        'wickets': ['sum', 'mean', 'count'],
        'total': 'mean'
    }).round(2)
    bowler_stats.columns = ['total_wickets', 'avg_wickets', 'innings', 'avg_team_total_conceded']
    bowler_stats = bowler_stats.sort_values('total_wickets', ascending=False)
    
    print("\nTop 10 Bowlers by Total Wickets:")
    print(bowler_stats.head(10))
    
    # 7. Feature engineering (same as training pipeline)
    print("\n" + "="*50)
    print("6. FEATURE ENGINEERING")
    print("="*50)
    
    df_engineered = df.copy()
    
    # Run rate
    df_engineered['run_rate'] = df_engineered['runs'] / (df_engineered['overs'] + 1e-5)
    
    # Required run rate
    df_engineered['required_run_rate'] = (df_engineered['total'] - df_engineered['runs']) / (20 - df_engineered['overs'] + 1e-5)
    
    # Wickets remaining
    df_engineered['wickets_remaining'] = 10 - df_engineered['wickets']
    
    # Overs remaining
    df_engineered['overs_remaining'] = 20 - df_engineered['overs']
    
    # Match phase
    def get_match_phase(overs):
        if overs <= 6:
            return 'Powerplay'
        elif overs <= 15:
            return 'Middle'
        else:
            return 'Death'
    
    df_engineered['match_phase'] = df_engineered['overs'].apply(get_match_phase)
    
    # Team batting strength
    team_avg_runs = df_engineered.groupby('bat_team')['runs'].mean()
    df_engineered['team_batting_strength'] = df_engineered['bat_team'].map(team_avg_runs)
    
    # Team bowling strength
    team_avg_wickets = df_engineered.groupby('bowl_team')['wickets'].mean()
    df_engineered['team_bowling_strength'] = df_engineered['bowl_team'].map(team_avg_wickets)
    
    print("Engineered Features Sample:")
    print(df_engineered[['run_rate', 'required_run_rate', 'wickets_remaining', 'overs_remaining', 
                        'match_phase', 'team_batting_strength', 'team_bowling_strength']].head())
    
    # 8. Correlation analysis
    print("\n" + "="*50)
    print("7. CORRELATION ANALYSIS")
    print("="*50)
    
    # Prepare data for correlation (same encoding as training)
    df_corr = df_engineered.copy()
    categorical_cols = ['bat_team', 'bowl_team', 'venue', 'batsman', 'bowler', 'match_phase']
    
    for col in categorical_cols:
        if col in df_corr.columns:
            le = LabelEncoder()
            df_corr[col] = le.fit_transform(df_corr[col])
    
    # Drop non-numeric columns
    df_corr = df_corr.drop(['mid', 'date'], axis=1, errors='ignore')
    
    # Correlation with target
    correlation_matrix = df_corr.corr()
    correlations_with_total = correlation_matrix['total'].abs().sort_values(ascending=False)
    
    print("Top 10 Features Correlated with Total Score:")
    for feature, corr in correlations_with_total.head(10).items():
        print(f"  {feature}: {corr:.3f}")
    
    # 9. Key insights
    print("\n" + "="*60)
    print("KEY INSIGHTS AND RECOMMENDATIONS")
    print("="*60)
    
    print("\n1. DATA QUALITY:")
    print(f"   - Original dataset shape: {df.shape}")
    print(f"   - Missing values: {df.isnull().sum().sum()}")
    print(f"   - Duplicate rows: {df.duplicated().sum()}")
    
    print("\n2. TEAM PERFORMANCE:")
    print(f"   - Best batting team: {batting_stats.index[0]} (avg: {batting_stats.iloc[0]['avg_total']:.2f})")
    print(f"   - Best bowling team: {bowling_stats.index[0]} (avg conceded: {bowling_stats.iloc[0]['avg_total_conceded']:.2f})")
    
    print("\n3. VENUE INSIGHTS:")
    print(f"   - Highest scoring venue: {venue_stats.index[0]} (avg: {venue_stats.iloc[0]['avg_total']:.2f})")
    print(f"   - Number of venues: {df['venue'].nunique()}")
    
    print("\n4. PLAYER INSIGHTS:")
    print(f"   - Top batsman: {batsman_stats.index[0]} (total runs: {batsman_stats.iloc[0]['total_runs']:.0f})")
    print(f"   - Top bowler: {bowler_stats.index[0]} (total wickets: {bowler_stats.iloc[0]['total_wickets']:.0f})")
    
    print("\n5. FEATURE ENGINEERING RECOMMENDATIONS:")
    print("   - Add run_rate and required_run_rate")
    print("   - Add match_phase (Powerplay/Middle/Death)")
    print("   - Add team batting/bowling strength")
    print("   - Consider venue-specific statistics")
    print("   - Add player form/performance metrics")
    
    print("\n6. MODEL IMPROVEMENT SUGGESTIONS:")
    print("   - Use engineered features in training")
    print("   - Consider ensemble methods")
    print("   - Add feature selection")
    print("   - Implement cross-validation")
    print("   - Consider time-based splits")
    
    print("\n7. PIPELINE CONSISTENCY:")
    print("   - This EDA uses the same data loading and cleaning as training")
    print("   - Any changes to utils.py will be reflected in both EDA and training")
    print("   - Ensures consistency across the entire project")
    
    print("\n" + "="*60)
    print("EDA COMPLETED!")
    print("="*60)

if __name__ == "__main__":
    main() 