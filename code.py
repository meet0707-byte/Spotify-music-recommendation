"""
Spotify Clustering + Recommendation System
Modules: numpy, pandas, matplotlib, seaborn, scikit-learn
All plots are saved as PNGs in the 'plots' folder next to the CSV.
"""
import os
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity

# ------------------------------------------------------------------ CONFIG
CSV_PATH = r"upload_29247127-d163-4eee-859b-97c45b290960.csv"      
SHOW_PLOTS = False                    # True: windows pop up (close each to continue). False: only save PNGs.
K = 6                                # number of clusters: change this number by hand
AUDIO = ["danceability", "energy", "loudness", "speechiness", "acousticness",
         "instrumentalness", "liveness", "valence", "tempo", "duration_ms"]
# -------------------------------------------------------------------------

if not SHOW_PLOTS:
    matplotlib.use("Agg")
sns.set_theme(style="whitegrid")
plt.rcParams["figure.constrained_layout.use"] = True   # auto-fix overlapping titles/labels, also when you resize the window
OUT = os.path.join(os.path.dirname(os.path.abspath(CSV_PATH)), "plots")
os.makedirs(OUT, exist_ok=True)

def finish(name):
    plt.savefig(os.path.join(OUT, name + ".png"), dpi=120)
    if SHOW_PLOTS:
        plt.show()
    plt.close()

# ================================================== 1. PRE-PROCESSING
print("=" * 60, "\n1. PRE-PROCESSING\n" + "=" * 60)
raw = pd.read_csv(CSV_PATH)
print("Shape:", raw.shape)
print("\nMissing values:\n", raw.isna().sum()[raw.isna().sum() > 0])
print("Duplicate rows:", raw.duplicated().sum())
print("Duplicate track_ids (same song in several playlists):", raw.track_id.duplicated().sum())

df = raw.dropna(subset=["track_name", "track_artist"]).copy()
df["track_album_release_date"] = pd.to_datetime(df["track_album_release_date"], errors="coerce")
df["release_year"] = df["track_album_release_date"].dt.year
df["duration_min"] = df["duration_ms"] / 60000

# Unique songs table is used for clustering/recommendation so one song isn't counted many times
songs = (df.drop_duplicates(subset="track_id")
           .drop_duplicates(subset=["track_name", "track_artist"])
           .reset_index(drop=True))
print("Rows after cleaning:", len(df), "| unique songs:", len(songs))

# Outliers: clip extreme values (1st-99th percentile) for features with long tails
X_raw = songs[AUDIO].copy()
for c in ["speechiness", "liveness", "duration_ms", "tempo", "loudness"]:
    lo, hi = X_raw[c].quantile([0.01, 0.99])
    X_raw[c] = X_raw[c].clip(lo, hi)
X = StandardScaler().fit_transform(X_raw)       # scaling
print("Features scaled with StandardScaler:", AUDIO)

# ================================================== 2. EDA & VISUALS
print("\n" + "=" * 60, "\n2. DATA ANALYSIS & VISUALIZATION\n" + "=" * 60)
print(songs[AUDIO + ["track_popularity"]].describe().T.round(3))

# 2a distributions
fig, axes = plt.subplots(3, 4, figsize=(14, 8))
for ax, c in zip(axes.ravel(), AUDIO + ["track_popularity"]):
    sns.histplot(songs[c], kde=True, ax=ax, color="#1DB954")
    ax.set_title(c)
for ax in axes.ravel()[len(AUDIO) + 1:]:
    ax.axis("off")
finish("01_feature_distributions")

# 2b genre / subgenre counts
fig, axes = plt.subplots(1, 2, figsize=(13, 6))
sns.countplot(data=df, y="playlist_genre", order=df.playlist_genre.value_counts().index, ax=axes[0], color="#1DB954")
axes[0].set_title("Tracks per playlist genre")
sns.countplot(data=df, y="playlist_subgenre", order=df.playlist_subgenre.value_counts().index, ax=axes[1], color="#191414")
axes[1].set_title("Tracks per playlist subgenre")
finish("02_genre_counts")

# 2c top playlists
plt.figure(figsize=(9, 6))
df.playlist_name.value_counts().head(20).sort_values().plot.barh(color="#1DB954")
plt.title("Top 20 playlists by number of tracks")
finish("03_top_playlists")

# 2d top artists
plt.figure(figsize=(9, 6))
df.drop_duplicates("track_id").track_artist.value_counts().head(20).sort_values().plot.barh(color="#191414")
plt.title("Top 20 artists by number of unique songs")
finish("04_top_artists")

# 2e boxplots of features by genre
fig, axes = plt.subplots(3, 4, figsize=(14, 8))
for ax, c in zip(axes.ravel(), ["danceability", "energy", "loudness", "speechiness", "acousticness",
                                "instrumentalness", "liveness", "valence", "tempo", "duration_min",
                                "track_popularity"]):
    sns.boxplot(data=df, x="playlist_genre", y=c, ax=ax, hue="playlist_genre", legend=False, palette="viridis")
    ax.set_title(c); ax.set_xlabel(""); ax.tick_params(axis="x", rotation=30)
axes.ravel()[-1].axis("off")
finish("05_boxplots_by_genre")

# 2f mean audio profile per genre (heatmap)
prof = df.groupby("playlist_genre")[["danceability", "energy", "speechiness", "acousticness",
                                     "instrumentalness", "liveness", "valence"]].mean()
plt.figure(figsize=(10, 5))
sns.heatmap(prof, annot=True, fmt=".2f", cmap="YlGn")
plt.title("Average audio profile per genre")
finish("06_genre_profile_heatmap")

# 2g popularity by genre + violin
plt.figure(figsize=(10, 6))
sns.violinplot(data=df, x="playlist_genre", y="track_popularity", hue="playlist_genre", legend=False, palette="Set2")
plt.title("Popularity by genre")
finish("07_popularity_by_genre")

# 2h releases over time & energy/valence trend
yr = songs[songs.release_year >= 1980]
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
yr.groupby("release_year").size().plot(ax=axes[0], color="#1DB954")
axes[0].set_title("Songs by release year")
yr.groupby("release_year")[["energy", "valence", "danceability", "acousticness"]].mean().plot(ax=axes[1])
axes[1].set_title("Average audio features over time")
finish("08_time_trends")

# 2i key & mode
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
sns.countplot(data=songs, x="key", ax=axes[0], color="#1DB954"); axes[0].set_title("Musical key")
sns.countplot(data=songs, x="mode", ax=axes[1], color="#191414"); axes[1].set_title("Mode (0 = minor, 1 = major)")
finish("09_key_mode")

# 2j scatter relationships
fig, axes = plt.subplots(1, 3, figsize=(14, 5))
s = songs.sample(4000, random_state=1)
sns.scatterplot(data=s, x="energy", y="loudness", hue="playlist_genre", alpha=.6, ax=axes[0])
sns.scatterplot(data=s, x="energy", y="acousticness", hue="playlist_genre", alpha=.6, ax=axes[1], legend=False)
sns.scatterplot(data=s, x="danceability", y="valence", hue="playlist_genre", alpha=.6, ax=axes[2], legend=False)
finish("10_scatter_relationships")

# 2k pairplot (sample)
pp = sns.pairplot(df.sample(1500, random_state=1),
                  vars=["danceability", "energy", "valence", "acousticness", "speechiness"],
                  hue="playlist_genre", corner=True, plot_kws={"alpha": .5, "s": 15})
pp.savefig(os.path.join(OUT, "11_pairplot.png"), dpi=100)
if SHOW_PLOTS: plt.show()
plt.close("all")

# ================================================== 3. CORRELATION MATRIX
print("\n" + "=" * 60, "\n3. CORRELATION MATRIX\n" + "=" * 60)
num_cols = ["track_popularity"] + AUDIO + ["key", "mode"]
corr = songs[num_cols].corr()
print(corr.round(2))
plt.figure(figsize=(10, 8))
mask = np.triu(np.ones_like(corr, dtype=bool))   # hides upper triangle + diagonal in the stack below
sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap="coolwarm", center=0, square=True)
plt.title("Correlation matrix of features")
finish("12_correlation_matrix")
pairs = corr.where(~np.triu(np.ones_like(corr, dtype=bool), k=0)).stack().dropna().sort_values()
print("\nStrongest negative correlations:\n", pairs.head(3).round(2))
print("Strongest positive correlations:\n", pairs.tail(3).round(2))

# ================================================== 4. CLUSTERING
print("\n" + "=" * 60, "\n4. CLUSTERING (K-Means)\n" + "=" * 60)
# Elbow plot (optional, just to help you justify your choice of K)
inertia = [KMeans(n_clusters=k, n_init=10, random_state=42).fit(X).inertia_ for k in range(2, 11)]
plt.figure(figsize=(7, 4))
plt.plot(range(2, 11), inertia, "o-", color="#1DB954")
plt.title("Elbow method"); plt.xlabel("k"); plt.ylabel("inertia")
finish("13_elbow")
print("Using k =", K)

km = KMeans(n_clusters=K, n_init=20, random_state=42).fit(X)
songs["cluster"] = km.labels_      # `songs` is the main DataFrame: cleaned data + cluster column
print("\nCluster sizes:\n", songs.cluster.value_counts().sort_index())

# cluster profiles
cprof = songs.groupby("cluster")[AUDIO[:-1] + ["track_popularity"]].mean()
print("\nCluster profiles (means):\n", cprof.round(2))
plt.figure(figsize=(12, 5))
sns.heatmap((cprof - cprof.mean()) / cprof.std(), annot=cprof.round(2), fmt="", cmap="RdYlGn", center=0)
plt.title("Cluster profiles (colour = relative to other clusters, numbers = real mean)")
finish("14_cluster_profiles")

# PCA 2-D map
pca = PCA(n_components=2, random_state=42)
P = pca.fit_transform(X)
songs["pc1"], songs["pc2"] = P[:, 0], P[:, 1]
print("PCA explained variance:", pca.explained_variance_ratio_.round(3))

fig, axes = plt.subplots(1, 2, figsize=(14, 6))
sns.scatterplot(data=songs, x="pc1", y="pc2", hue="cluster", palette="tab10", s=8, alpha=.6, ax=axes[0])
axes[0].set_title("Clusters in PCA space")
sns.scatterplot(data=songs, x="pc1", y="pc2", hue="playlist_genre", palette="Set2", s=8, alpha=.6, ax=axes[1])
axes[1].set_title("Same space coloured by playlist genre")
finish("15_pca_clusters_vs_genre")

# subgenres
plt.figure(figsize=(9, 6))
sns.scatterplot(data=songs, x="pc1", y="pc2", hue="playlist_subgenre", s=8, alpha=.6, palette="tab20")
plt.legend(bbox_to_anchor=(1.01, 1), loc="upper left", fontsize=8, markerscale=2)
plt.title("PCA space coloured by playlist subgenre")
finish("16_pca_subgenre")

# playlist-name clusters: top 10 playlists coloured in PCA space
top_pl = songs.playlist_name.value_counts().head(10).index
sub = songs[songs.playlist_name.isin(top_pl)]
plt.figure(figsize=(9, 6))
sns.scatterplot(data=sub, x="pc1", y="pc2", hue="playlist_name", s=14, alpha=.7, palette="tab10")
plt.legend(bbox_to_anchor=(1.01, 1), loc="upper left", fontsize=8)
plt.title("Top 10 playlists in PCA space")
finish("17_pca_playlist_names")

# playlist-level clustering: each playlist = mean of its songs, clustered
pl = df.groupby("playlist_name")[AUDIO].mean()
pl = pl[df.groupby("playlist_name").size() >= 20]
PX = StandardScaler().fit_transform(pl)
pl["pl_cluster"] = KMeans(n_clusters=5, n_init=10, random_state=42).fit_predict(PX)
pp_ = PCA(2, random_state=42).fit_transform(PX)
plt.figure(figsize=(9, 6))
plt.scatter(pp_[:, 0], pp_[:, 1], c=pl.pl_cluster, cmap="tab10", s=40)
for i, name in enumerate(pl.index):
    if i % 4 == 0: plt.annotate(name[:22], (pp_[i, 0], pp_[i, 1]), fontsize=6)
plt.title("Playlists clustered by their average audio features")
finish("18_playlist_level_clusters")

# cluster vs genre cross-tab
ct = pd.crosstab(songs.cluster, songs.playlist_genre, normalize="index") * 100
print("\n% of each cluster by playlist genre:\n", ct.round(1))
plt.figure(figsize=(10, 5))
sns.heatmap(ct, annot=True, fmt=".0f", cmap="YlGnBu")
plt.title("Cluster composition by genre (% of cluster)")
finish("19_cluster_vs_genre")

# radar chart of clusters
cols = ["danceability", "energy", "speechiness", "acousticness", "instrumentalness", "liveness", "valence"]
ang = np.linspace(0, 2 * np.pi, len(cols), endpoint=False).tolist(); ang += ang[:1]
plt.figure(figsize=(8, 8)); ax = plt.subplot(polar=True)
for c in range(K):
    v = songs[songs.cluster == c][cols].mean().tolist(); v += v[:1]
    ax.plot(ang, v, label=f"Cluster {c}"); ax.fill(ang, v, alpha=.08)
ax.set_xticks(ang[:-1]); ax.set_xticklabels(cols); ax.legend(bbox_to_anchor=(1.2, 1))
plt.title("Cluster radar chart")
finish("20_cluster_radar")

# ================================================== 5. RECOMMENDER
print("\n" + "=" * 60, "\n5. RECOMMENDATION SYSTEM\n" + "=" * 60)

def find_song(query):
    m = songs[songs.track_name.str.lower().str.contains(query.lower(), regex=False, na=False)]
    return m.sort_values("track_popularity", ascending=False)

def recommend(song_name, n=10, same_cluster=True, artist=None):
    """Top-n songs most similar (cosine similarity on scaled audio features)."""
    m = find_song(song_name)
    if artist:
        m = m[m.track_artist.str.lower().str.contains(artist.lower(), regex=False)]
    if m.empty:
        print(f"'{song_name}' not found."); return None
    i = m.index[0]
    cand = np.where(songs.cluster == songs.at[i, "cluster"])[0] if same_cluster else np.arange(len(songs))
    sims = cosine_similarity(X[i].reshape(1, -1), X[cand]).ravel()
    out = songs.iloc[cand].copy(); out["similarity"] = sims
    out = out[(out.index != i) & ~((out.track_name == songs.at[i, "track_name"]) &
                                    (out.track_artist == songs.at[i, "track_artist"]))]
    out = out.sort_values("similarity", ascending=False).head(n)
    print(f"\nBecause you like: {songs.at[i,'track_name']} - {songs.at[i,'track_artist']} "
          f"[{songs.at[i,'playlist_genre']}, cluster {songs.at[i,'cluster']}]")
    print(out[["track_name", "track_artist", "playlist_genre", "cluster", "similarity"]]
          .round(3).to_string(index=False))
    return out

def recommend_for_artist(artist, n=10):
    """Recommend based on the average taste profile of an artist's songs."""
    idx = songs.index[songs.track_artist.str.lower() == artist.lower()]
    if len(idx) == 0:
        print(f"Artist '{artist}' not found."); return None
    taste = X[idx].mean(axis=0).reshape(1, -1)
    sims = cosine_similarity(taste, X).ravel()
    out = songs.copy(); out["similarity"] = sims
    out = out[~out.index.isin(idx)].sort_values("similarity", ascending=False).head(n)
    print(f"\nBecause you like the artist: {artist}")
    print(out[["track_name", "track_artist", "playlist_genre", "similarity"]].round(3).to_string(index=False))
    return out

def recommend_for_playlist(playlist_name, n=10):
    """Recommend songs matching the average sound of a playlist."""
    idx = songs.index[songs.playlist_name.str.lower() == playlist_name.lower()]
    if len(idx) == 0:
        print(f"Playlist '{playlist_name}' not found."); return None
    taste = X[idx].mean(axis=0).reshape(1, -1)
    out = songs.copy(); out["similarity"] = cosine_similarity(taste, X).ravel()
    out = out[~out.index.isin(idx)].sort_values("similarity", ascending=False).head(n)
    print(f"\nSongs similar to the playlist: {playlist_name}")
    print(out[["track_name", "track_artist", "playlist_genre", "similarity"]].round(3).to_string(index=False))
    return out

# ---- demo (edit these to try your own songs) ----
demo = songs.sort_values("track_popularity", ascending=False).iloc[0]
recommend(demo.track_name, n=10)
recommend_for_artist("Ed Sheeran", n=10)
recommend_for_playlist("Pop Remix", n=10)

# quick sanity check of the system: do recommendations stay in the same genre?
rng = np.random.RandomState(0); hit = []
for i in rng.choice(len(songs), 300, replace=False):
    cand = np.where(songs.cluster == songs.at[i, "cluster"])[0]
    sims = cosine_similarity(X[i].reshape(1, -1), X[cand]).ravel()
    top = cand[np.argsort(-sims)[1:11]]
    hit.append((songs.loc[top, "playlist_genre"] == songs.at[i, "playlist_genre"]).mean())
print(f"\nEvaluation: on average {np.mean(hit)*100:.1f}% of the top-10 recommendations share the "
      f"query song's playlist genre (random chance would be ~{(songs.playlist_genre.value_counts(normalize=True)**2).sum()*100:.0f}%).")

songs.to_csv(os.path.join(OUT, "songs_with_clusters.csv"), index=False)
print("\nDone. Plots and songs_with_clusters.csv saved in:", OUT)

# ---- interactive loop: type a song and get recommendations ----
while True:
    q = input("\nEnter a song name (or press Enter to quit): ").strip()
    if not q:
        break
    recommend(q, n=10)
