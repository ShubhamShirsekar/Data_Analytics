# Bitcoin Investment Strategy Analysis: DCA vs Buy the Dip

## Overview
A comprehensive quantitative analysis comparing two popular Bitcoin investment strategies using historical data from 2012 to 2025. This project evaluates Dollar Cost Averaging (DCA) against a "Buy the Dip" strategy using risk-adjusted return metrics across 3,950 rolling 3-year windows.

## Research Question
Which strategy yields higher risk-adjusted returns (Sharpe Ratio) over rolling 3-year periods: Dollar Cost Averaging or Buy the Dip?

## Methodology

### Strategies Compared

**Dollar Cost Averaging (DCA)**
- Fixed investment: $1,000 per month
- Investment timing: First US trading day of each month
- Consistent buying regardless of price

**Buy the Dip**
- Triggers when price drops more than 10% from 30-day rolling high
- Equal total capital as DCA, split across all dip signals
- Opportunistic buying during market corrections

### Data Source
Bitcoin historical price data from Kaggle: [Bitcoin Historical Data](https://www.kaggle.com/datasets/mczielinski/bitcoin-historical-data)
- Time period: 2012-2025
- Granularity: Daily OHLCV data
- Exchange: Bitstamp USD

### Analysis Framework
1. Data preparation and validation
2. Strategy definition and parameter setting
3. Backtesting across rolling 3-year windows
4. Performance metrics calculation (returns, volatility, Sharpe Ratio)
5. Comprehensive visualization and comparison

## Key Findings

### Performance Metrics
- **DCA Average Return**: 312.63% (total), annualized performance calculated
- **Buy the Dip Average Return**: 268.72% (total), annualized performance calculated
- **DCA Win Rate**: 62.99% of all windows
- **Buy the Dip Win Rate**: 37.01% of all windows

### Risk-Adjusted Returns
- **DCA Average Sharpe Ratio**: Superior risk-adjusted performance
- **Strategy Conclusion**: DCA provides more consistent risk-adjusted returns across market cycles
- **Market Context**: Buy the Dip can outperform during specific market conditions but with higher volatility

## Project Structure
```
├── data_preparation.py          # Data loading and cleaning
├── strategy_definition.py       # DCA and Buy the Dip implementation
├── backtesting_framework.py     # Rolling window simulation
├── performance_metrics.py       # Sharpe Ratio and risk calculations
├── visualizations.py            # Comprehensive charts and analysis
└── README.md
```

## Visualizations
The project includes three comprehensive visualization suites:

1. **Strategy Performance Comparison**
   - Side-by-side metric comparisons
   - Returns vs volatility over time
   - Risk-return scatter plots

2. **Rolling Window Analysis**
   - Sharpe Ratio evolution timeline
   - Year-by-year performance heatmap
   - Market condition breakdowns

3. **Portfolio Value Timeline**
   - Portfolio growth trajectories
   - Bitcoin accumulation curves
   - Break-even point identification

## Requirements
```
pandas
numpy
matplotlib
seaborn
opendatasets
kaggle
```

## Usage

### 1. Data Download
```python
# Using opendatasets
import opendatasets as od
od.download('https://www.kaggle.com/datasets/mczielinski/bitcoin-historical-data')
```

### 2. Run Analysis
```python
# Execute notebooks in order:
# 1. data_preparation.py
# 2. strategy_definition.py
# 3. backtesting_framework.py
# 4. performance_metrics.py
# 5. visualizations.py
```

### 3. Generate Visualizations
All visualizations are automatically saved as high-resolution PNG files suitable for presentations and reports.

## Key Assumptions
- Risk-free rate: 2% annually (historical US Treasury average)
- No transaction fees or slippage considered
- US federal holidays observed for DCA purchases
- 30-day lookback period for dip detection
- Equal total capital allocation between strategies

## Limitations
- Historical backtesting does not guarantee future performance
- Does not account for transaction costs or taxes
- Simplified volatility calculations for computational efficiency
- Limited to Bitcoin; results may not generalize to other assets

## Results Interpretation
The analysis demonstrates that while both strategies can generate substantial returns in cryptocurrency markets, DCA provides superior risk-adjusted performance with more consistent results across different market conditions. Buy the Dip strategies may appeal to investors willing to accept higher volatility for potential outperformance during specific market regimes.

## License
MIT License

## Author
Shubham Shirsekar

## Acknowledgments
- Bitcoin historical data provided by Kaggle user mczielinski
- Analysis inspired by academic research on systematic investment strategies
