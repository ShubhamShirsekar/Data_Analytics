# 🏏 IPL Ball-by-Ball Data Analytics

A comprehensive data analytics project exploring Indian Premier League (IPL) cricket data using Python, focusing on real-world data science applications and insights.

## 📊 Project Overview

This project analyzes **260,921+ real IPL deliveries** from the Kaggle dataset, covering multiple seasons of IPL cricket. The analysis includes player performance, team strategies, match dynamics, and statistical insights from ball-by-ball data.

## 🎯 Learning Objectives

This project is part of a **Masters in Data Analytics** curriculum, covering:

- **Data Formats & Loading**: Working with large CSV datasets
- **Data Modelling**: Understanding hierarchical sports data structures  
- **Descriptive Statistics**: Extracting meaningful insights from sports data
- **Advanced Queries**: Complex filtering and aggregation for cricket analytics
- **Python Programming**: Pandas, NumPy, Matplotlib, Seaborn

## 🗂️ Dataset Information

### Data Source
**Primary Dataset**: [IPL Complete Dataset (2008-2020)](https://www.kaggle.com/datasets/patrickb1912/ipl-complete-dataset-20082020)
- **Creator**: [Patrick B](https://www.kaggle.com/patrickb1912)
- **License**: [Dataset License on Kaggle](https://www.kaggle.com/datasets/patrickb1912/ipl-complete-dataset-20082020)
- **Last Updated**: Check Kaggle page for latest updates

### Dataset Structure
The `deliveries.csv` contains ball-by-ball data with the following schema:

| Column | Type | Description |
|--------|------|-------------|
| `match_id` | int | Unique identifier for each match |
| `inning` | int | Innings number (1 or 2) |
| `batting_team` | str | Team currently batting |
| `bowling_team` | str | Team currently bowling |
| `over` | int | Over number (0-19 for T20) |
| `ball` | int | Ball number in the over |
| `batter` | str | Batsman currently on strike |
| `bowler` | str | Current bowler |
| `non_striker` | str | Non-striker batsman |
| `batsman_runs` | int | Runs scored by batsman |
| `extra_runs` | int | Extra runs (wides, byes, etc.) |
| `total_runs` | int | Total runs scored off this delivery |
| `extras_type` | str | Type of extra (if any) |
| `is_wicket` | int | Whether a wicket fell (1=Yes, 0=No) |
| `player_dismissed` | str | Name of dismissed player |
| `dismissal_kind` | str | Mode of dismissal |
| `fielder` | str | Fielder involved in dismissal |

## 🚀 Getting Started

### Prerequisites
```bash
pip install pandas numpy matplotlib seaborn
```

### Installation
1. Clone this repository:
```bash
git clone https://github.com/YOUR_USERNAME/ipl-analytics.git
cd ipl-analytics
```

2. Download the dataset:
   - Visit [Kaggle IPL Dataset](https://www.kaggle.com/datasets/patrickb1912/ipl-complete-dataset-20082020)
   - Download `deliveries.csv`
   - Place it in the project root directory

3. Run the analysis:
```bash
python ipl_analysis.py
```

## 📈 Key Insights & Analytics

### Player Performance
- **Top Run Scorers**: Analysis of highest run scorers across IPL history
- **Best Bowlers**: Most economical bowlers and leading wicket-takers
- **Strike Rate Analysis**: Batting efficiency metrics

### Team Analytics
- **PowerPlay Performance**: First 6 overs analysis by team
- **Death Over Specialists**: Performance in overs 17-20
- **Boundary Analysis**: 4s vs 6s patterns across teams

### Match Dynamics
- **Scoring Patterns**: Run distribution across different phases
- **Wicket Timing**: When wickets fall most frequently
- **Extras Analysis**: Impact of wides, no-balls, and other extras

## 🗃️ Project Structure

```
ipl-analytics/
│
├── README.md                 # Project documentation
├── ipl_analysis.py          # Main analysis script
├── requirements.txt         # Python dependencies
├── deliveries.csv          # Dataset (download separately)
├── LICENSE                 # Project license
└── .gitignore             # Git ignore file
```

## 📊 Sample Outputs

The analysis provides insights such as:
- Total deliveries analyzed: **260,921+**
- Boundary rate: **~8-10%** of all deliveries
- Wicket rate: **~3.5%** (1 wicket every ~29 balls)
- Most common dismissal: **Caught**
- PowerPlay vs Death overs scoring comparison

## 🔮 Future Enhancements

- [ ] **Data Visualization**: Interactive charts and dashboards
- [ ] **Time Series Analysis**: Performance trends across seasons
- [ ] **Machine Learning**: Match outcome prediction models
- [ ] **Player Network Analysis**: Batting partnerships and bowling combinations
- [ ] **Venue Analysis**: Ground-specific performance patterns

## 🤝 Contributing

This is a learning project, but contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📚 Learning Resources

- [Pandas Documentation](https://pandas.pydata.org/docs/)
- [Cricket Analytics with Python](https://www.analyticsvidhya.com/blog/2021/04/cricket-match-prediction-using-machine-learning/)
- [IPL Official Website](https://www.iplt20.com/)

## 🙏 Acknowledgments

- **Dataset Creator**: [Patrick B](https://www.kaggle.com/patrickb1912) for providing the comprehensive IPL dataset
- **Kaggle Community**: For maintaining and sharing cricket datasets
- **IPL Organization**: For creating such an exciting tournament that generates rich data
- **Tutorial Inspiration**: Part of Masters in Data Analytics curriculum

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

### Dataset License
Please refer to the [original Kaggle dataset license](https://www.kaggle.com/datasets/patrickb1912/ipl-complete-dataset-20082020) for data usage terms.

## 📞 Contact

**Student**: [Your Name]
**Course**: Masters in Data Analytics
**Email**: [your.email@example.com]
**LinkedIn**: [Your LinkedIn Profile]

---

⭐ **Star this repository if you found it helpful!**

📚 **Part of a comprehensive Data Analytics learning journey covering real-world applications**