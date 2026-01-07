# Feature Engineering Reference Guide

## 📊 Complete Feature List (110+ Features)

---

## 1️⃣ Original Audio Features (9)
These are the core Spotify audio features describing musical characteristics.

| Feature | Description | Range | Interpretation |
|---------|-------------|-------|----------------|
| **acousticness** | Confidence measure of acoustic music | 0-1 | 1.0 = definitely acoustic |
| **danceability** | How suitable for dancing | 0-1 | Higher = more danceable |
| **energy** | Intensity and activity measure | 0-1 | Higher = more energetic |
| **instrumentalness** | Predicts lack of vocals | 0-1 | >0.5 = likely instrumental |
| **liveness** | Presence of audience | 0-1 | >0.8 = likely live recording |
| **loudness** | Overall loudness in dB | -42 to 0 | Higher = louder |
| **speechiness** | Presence of spoken words | 0-1 | >0.66 = mostly speech |
| **tempo** | Overall tempo in BPM | 0-210 | Beats per minute |
| **valence** | Musical positiveness | 0-1 | Higher = more positive/happy |

---

## 2️⃣ Temporal Features (7)

### Engineered from Dates
| Feature | Description | Calculation | Use Case |
|---------|-------------|-------------|----------|
| **song_age** | Years since release | 2026 - release_year | Identify vintage vs recent songs |
| **years_since_top** | Years since top year | 2026 - top_year | Recency of user preference |
| **release_month** | Month of release | From release_date | Seasonal patterns |
| **release_day_of_year** | Day number in year | From release_date | Specific timing |
| **release_decade** | Decade of release | (release_year // 10) * 10 | Era classification |

### Original
- **release_year**: Year song was released
- **top_year**: Year song was in user's top playlist

---

## 3️⃣ Interaction Features (6)
Capture relationships between audio characteristics.

| Feature | Formula | Captures |
|---------|---------|----------|
| **energy_loudness** | energy × loudness | Overall intensity/power |
| **danceability_tempo** | danceability × tempo | Dance potential with rhythm |
| **valence_energy** | valence × energy | Happy and energetic songs |
| **acoustic_instrumental** | acousticness × instrumentalness | Organic instrumental music |
| **speech_to_music_ratio** | speechiness / (instrumentalness + 0.001) | Voice dominance |
| **live_to_studio_ratio** | liveness / (1 - liveness + 0.001) | Live performance quality |

---

## 4️⃣ Categorical Binary Features (13)

### Music Theory
| Feature | Description | Values |
|---------|-------------|--------|
| **mode_major** | Major key | 0 or 1 |
| **mode_minor** | Minor key | 0 or 1 |

### Song Characteristics
| Feature | Condition | Purpose |
|---------|-----------|---------|
| **is_short_song** | length < 3 minutes | Quick listens |
| **is_long_song** | length > 5 minutes | Extended pieces |
| **is_popular** | popularity > 50 | Mainstream songs |

### Energy Levels
| Feature | Condition | Classification |
|---------|-----------|----------------|
| **is_high_energy** | energy > 0.7 | High-intensity songs |
| **is_low_energy** | energy < 0.3 | Calm songs |

### Mood Indicators
| Feature | Condition | Interpretation |
|---------|-----------|----------------|
| **is_happy** | valence > 0.6 | Positive mood |
| **is_sad** | valence < 0.4 | Melancholic mood |

### Additional
- **length_minutes**: Song length in minutes (derived from milliseconds)
- Plus encoded versions of **key** and **time_signature**

---

## 5️⃣ Text Features (TF-IDF Encoded - 110 features)

### Artist TF-IDF Features (50)
- **artist_tfidf_0** through **artist_tfidf_49**
- Captures artist name patterns and genre associations
- Identifies similar artists based on naming conventions
- Helps recommend songs from artists with similar styles

### Song Name TF-IDF Features (30)
- **song_tfidf_0** through **song_tfidf_29**
- Captures common words/themes in song titles
- Identifies songs with similar naming patterns
- Useful for thematic recommendations

### Album TF-IDF Features (30)
- **album_tfidf_0** through **album_tfidf_29**
- Captures album naming patterns
- Identifies related album collections
- Groups songs from similar album contexts

**TF-IDF Parameters:**
- max_features: Limited to most important terms
- min_df=2: Words must appear in at least 2 documents
- Automatically captures multi-word patterns (bigrams for artists)

---

## 6️⃣ Original Categorical Features (3)

| Feature | Description | Values |
|---------|-------------|--------|
| **key** | Musical key (pitch class) | 0-11 (C, C#, D, etc.) |
| **mode** | Major or minor scale | 0 (minor) or 1 (major) |
| **time_signature** | Beats per measure | 0, 1, 3, 4, 5 |

---

## 📈 Feature Selection Strategy

### For Recommendation Model
The following features are used in the final model:

```python
audio_features = [
    'acousticness', 'danceability', 'energy', 'instrumentalness',
    'liveness', 'loudness', 'speechiness', 'tempo', 'valence'
]

temporal_features = ['song_age', 'years_since_top', 'release_month']

interaction_features = [
    'energy_loudness', 'danceability_tempo', 'valence_energy',
    'acoustic_instrumental', 'speech_to_music_ratio'
]

categorical_binary = [
    'mode_major', 'is_popular', 'is_high_energy',
    'is_happy', 'is_short_song', 'is_long_song'
]

text_features = [all TF-IDF features from artist, song, album]
```

**Total Model Features**: 110+

---

## 🎯 Why Each Feature Type Matters

### Audio Features
✅ Core musical characteristics  
✅ Objectively measurable  
✅ Consistent across platforms  

### Text Features (TF-IDF)
✅ Capture artist/genre preferences  
✅ Identify naming patterns  
✅ Essential for style matching  

### Temporal Features
✅ Account for song age preferences  
✅ Capture era-specific tastes  
✅ Seasonal patterns  

### Interaction Features
✅ Reveal hidden relationships  
✅ Capture complex preferences  
✅ Non-linear patterns  

### Categorical Features
✅ Music theory elements  
✅ Binary classifications  
✅ Discrete characteristics  

---

## 🔧 Feature Scaling Strategy

### StandardScaler Applied To:
- All audio features (9)
- All temporal features (3)
- All interaction features (5)

### Not Scaled:
- TF-IDF features (already normalized)
- Binary features (0 or 1)
- Categorical encoded features

---

## 💡 Feature Engineering Best Practices

1. **Domain Knowledge**: Audio features based on music theory
2. **Interaction Terms**: Capture feature combinations
3. **Text Encoding**: TF-IDF for artist/song patterns
4. **Temporal Context**: Account for time-based patterns
5. **Normalization**: Scale features appropriately
6. **Feature Selection**: Use all relevant information

---

## 📊 Feature Importance Insights

Based on correlation analysis:

**Highly Correlated Features:**
- energy ↔ loudness (positive)
- acousticness ↔ energy (negative)
- valence ↔ danceability (positive)

**Unique Information:**
- Text features: Artist/genre preferences
- Temporal features: Era preferences
- Interaction features: Complex tastes

---

## 🚀 Using Features in Practice

### For User Profiling
1. Calculate average feature values per user
2. Identify distinctive characteristics
3. Create user "taste signatures"

### For Recommendations
1. Train model on user's feature space
2. Score candidates using all features
3. Rank by similarity to user profile

### For Analysis
1. Visualize feature distributions
2. Identify correlations
3. Understand user preferences

---

**Total Feature Count**: 110+ comprehensive features  
**Original Feature Count**: 9 basic audio features  
**Improvement Factor**: 12x more information for modeling
