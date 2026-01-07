# How to Use the Spotify Recommendation Notebook

## 🚀 Quick Start Guide

---

## Prerequisites

### Required Files
- ✅ `Spotify_project_Task2_v2.ipynb` - The enhanced notebook
- ✅ `reconstructed_df.csv` - Your dataset (3,591 songs)

### Required Libraries
All libraries are automatically imported in the notebook. If any are missing, they will be installed automatically.

**Main libraries:**
- pandas, numpy (data manipulation)
- scikit-learn (machine learning)
- matplotlib, seaborn (visualization)
- scipy (statistical analysis)

---

## 📋 Step-by-Step Instructions

### Step 1: Open the Notebook
1. Open `Spotify_project_Task2_v2.ipynb` in Jupyter Notebook or VS Code
2. Ensure `reconstructed_df.csv` is in the same directory
3. Select Python 3.11+ kernel

### Step 2: Run All Cells Sequentially
Execute cells from top to bottom. The notebook is organized in 9 main sections:

#### Section 1: Data Exploration (Cells 1-4)
- Imports all libraries
- Loads the dataset
- Shows available features
- Performs data quality checks
- **What to expect**: Summary of 3,591 songs, 5 users, feature lists

#### Section 2: Feature Engineering (Cells 5-7)
- Creates temporal features (song age, etc.)
- Encodes categorical features
- Generates interaction features
- Encodes text features with TF-IDF
- **What to expect**: 110+ features created, processing messages

#### Section 3: Correlation Analysis (Cells 8-9)
- Generates correlation heatmaps
- Identifies strong correlations
- Visualizes feature relationships
- **What to expect**: Colorful heatmaps, correlation insights

#### Section 4: Feature Preparation (Cell 10)
- Selects features for modeling
- Scales numerical features
- **What to expect**: Feature summary, scaling confirmation

#### Section 5: Enhanced Model (Cells 11-12)
- Defines enhanced recommendation function
- Tests with sample user
- **What to expect**: Function definition, sample recommendations

#### Section 6: Model Evaluation (Cells 13-14)
- Evaluates model quality
- Calculates recall metrics
- Visualizes performance
- **What to expect**: Performance metrics, bar charts, recall scores

#### Section 7: Generate Recommendations (Cell 15)
- Creates complete playlists for all users
- Ranks all songs by relevance
- **What to expect**: 5 complete playlists, one per user

#### Section 8: Feature Importance (Cell 16)
- Analyzes user taste profiles
- Visualizes feature distributions
- **What to expect**: Bar charts showing user preferences

#### Section 9: Summary (Cell 17)
- Compares original vs enhanced model
- Shows key improvements
- **What to expect**: Comprehensive summary

---

## 🎯 Key Outputs

### 1. Correlation Heatmaps
Location: Section 3  
**Shows**: Relationships between features  
**Use**: Understand which features correlate

### 2. User Taste Profiles
Location: Section 8  
**Shows**: Average feature values per user  
**Use**: Understand user preferences

### 3. Recommendation Playlists
Location: Section 7  
**Shows**: Ranked songs for each user  
**Structure**:
- Rank 0: User's existing library
- Rank 1+: Recommended songs (best to worst)

### 4. Evaluation Metrics
Location: Section 6  
**Shows**: Model performance per user  
**Metrics**: Recall (% of user songs correctly identified)

---

## 💡 Understanding the Results

### Sample Recommendation Output

```
name                 artist              score    original_owner
"Song Name"          "Artist Name"       0.85     gamma
```

**Columns Explained:**
- **name**: Song title
- **artist**: Artist name
- **score**: Similarity score (higher = better match)
- **original_owner**: Which user originally owned this song
- **rank**: Position in recommendation list

### Interpreting Scores
- **score > 0**: Song matches user's taste (inside boundary)
- **score < 0**: Song doesn't match (outside boundary)
- **Higher score = Better match**

### Recall Metric
- **Recall = 0.90**: Model correctly identifies 90% of user's songs
- **Higher is better**
- Values typically range from 0.70 to 0.95

---

## 🔧 Customization Options

### Change Number of Recommendations
In cell with `recommend_for_user_enhanced()`:
```python
# Change top_n parameter
recommendations = recommend_for_user_enhanced(
    target_user='alpha',
    data=df_scaled,
    features=all_model_features,
    top_n=20  # Change from 10 to 20
)
```

### Change Model Parameters
In the model definition:
```python
model = OneClassSVM(
    kernel='rbf',
    nu=0.05,      # Change to 0.10 for looser boundary
    gamma='scale'  # Or 'auto' for different sensitivity
)
```

### Test Different Users
```python
# Get all users
users = df_scaled['user'].unique()
print(users)  # ['alpha', 'beta', 'gamma', 'delta', 'epsilon']

# Test specific user
test_user = 'beta'  # Change this
recommendations = recommend_for_user_enhanced(test_user, df_scaled, all_model_features)
```

### Add More Features
In Section 2 (Feature Engineering), add your own:
```python
# Add custom feature
df_engineered['my_feature'] = df_engineered['energy'] + df_engineered['valence']

# Then add to feature list
all_model_features.append('my_feature')
```

---

## 📊 Visualization Customization

### Change Plot Colors
```python
# In heatmap
sns.heatmap(..., cmap='viridis')  # Try: 'coolwarm', 'RdYlGn', 'Blues'

# In bar plots
plt.bar(..., color='steelblue')  # Try: 'coral', 'lightgreen', 'skyblue'
```

### Adjust Figure Sizes
```python
plt.figure(figsize=(14, 10))  # (width, height) in inches
```

---

## 🐛 Troubleshooting

### "Module not found: seaborn"
**Solution**: Run in a cell:
```python
!pip install seaborn
```

### "File not found: reconstructed_df.csv"
**Solution**: 
1. Check file is in same directory as notebook
2. Or specify full path:
```python
df = pd.read_csv(r'C:\full\path\to\reconstructed_df.csv')
```

### Memory Error
**Solution**: Reduce TF-IDF features:
```python
# Change max_features
artist_vectorizer = TfidfVectorizer(max_features=30)  # Reduce from 50
```

### Kernel Crash
**Solution**: 
1. Restart kernel
2. Run cells again from top
3. Or clear output and restart fresh

---

## 📈 Expected Execution Time

- **Full notebook**: 2-5 minutes
- **Feature engineering**: 30-60 seconds
- **Model training**: 10-20 seconds per user
- **Visualization**: 5-10 seconds per plot

**Total cells**: 17 code cells + 9 markdown cells

---

## 💾 Saving Results

### Export Recommendations to CSV
Add this cell at the end:

```python
# Export recommendations for all users
for user, playlist in user_recommendation_playlists.items():
    filename = f'recommendations_{user}.csv'
    playlist.to_csv(filename, index=False)
    print(f"Saved: {filename}")
```

### Save Visualizations
After any plot:
```python
plt.savefig('correlation_heatmap.png', dpi=300, bbox_inches='tight')
```

---

## 🎓 Learning Path

### Beginner
1. Run all cells without modification
2. Observe outputs and visualizations
3. Read the comments in each cell

### Intermediate
1. Modify recommendation parameters
2. Test different users
3. Adjust visualization styles

### Advanced
1. Add custom features
2. Try different ML models
3. Implement ensemble methods
4. Experiment with feature selection

---

## 📝 Best Practices

✅ **Run cells sequentially** - Don't skip cells  
✅ **Check outputs** - Verify each step completes  
✅ **Read comments** - Understand what each cell does  
✅ **Save work** - Export results before closing  
✅ **Document changes** - Note any modifications  

❌ **Don't** skip feature engineering  
❌ **Don't** modify feature names without updating model  
❌ **Don't** run cells out of order  

---

## 🔍 What to Look For

### Good Signs ✅
- All cells execute without errors
- Heatmaps show clear patterns
- Recall scores > 0.70
- Recommendations make sense for user profiles

### Warning Signs ⚠️
- Many errors during execution
- All recall scores < 0.50
- Empty recommendation lists
- Missing visualizations

---

## 🎯 Success Criteria

Your notebook is working correctly if:

1. ✅ All 110+ features are created
2. ✅ Correlation heatmaps display properly
3. ✅ Average recall > 0.70
4. ✅ Each user gets ranked recommendations
5. ✅ Visualizations show distinct user profiles

---

## 📚 Additional Resources

### Understanding TF-IDF
- Captures word importance in text
- Higher value = more distinctive word
- Used for artist/song/album patterns

### Understanding One-Class SVM
- Learns boundary of user's taste
- Scores songs by distance to boundary
- Positive scores = inside boundary (good match)

### Understanding Correlation
- Range: -1 to +1
- Positive: Features increase together
- Negative: One increases, other decreases
- Zero: No linear relationship

---

## 🚀 Next Steps

After running the notebook:

1. **Analyze Results**: Look at user profiles and recommendations
2. **Compare Users**: See how tastes differ
3. **Validate Recommendations**: Check if they make musical sense
4. **Export Data**: Save playlists for presentation
5. **Document Insights**: Note interesting patterns

---

## 📞 Need Help?

Check these sections in order:
1. Error messages in notebook
2. Troubleshooting section above
3. Comments in notebook cells
4. Python/sklearn documentation

---

**Status**: Ready to Use  
**Difficulty**: Intermediate  
**Time Required**: 5 minutes to run, 30 minutes to understand fully  
**Output**: Complete recommendation system with analysis
