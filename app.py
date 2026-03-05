import streamlit as st
import pandas as pd
import torch
import torch.nn as nn
import pickle
from sklearn.preprocessing import LabelEncoder

# 1. NCF Class Definition
class NCF(nn.Module):
    def __init__(self, n_users, n_movies, n_factors=64, hidden_units=[128, 64, 32], dropout=0.2):
        super(NCF, self).__init__()
        self.user_embedding = nn.Embedding(n_users, n_factors)
        self.movie_embedding = nn.Embedding(n_movies, n_factors)
        self.fc_layers = nn.ModuleList()
        input_size = n_factors * 2
        for hidden_size in hidden_units:
            self.fc_layers.append(nn.Linear(input_size, hidden_size))
            self.fc_layers.append(nn.ReLU())
            self.fc_layers.append(nn.Dropout(dropout))
            input_size = hidden_size
        self.output_layer = nn.Linear(input_size, 1)

    def forward(self, user_ids, movie_ids):
        user_emb = self.user_embedding(user_ids)
        movie_emb = self.movie_embedding(movie_ids)
        x = torch.cat([user_emb, movie_emb], dim=1)
        for layer in self.fc_layers:
            x = layer(x)
        return self.output_layer(x).squeeze()

# 2. Load Resources & Define Encoders
@st.cache_resource
def load_resources():
    # Load data
    movies = pd.read_csv('data/movies.csv')
    ratings = pd.read_csv('data/ratings.csv')
    
    # Filter data to match the training subset
    min_movie_ratings = 20
    movie_counts = ratings['movieId'].value_counts()
    valid_movie_ids = movie_counts[movie_counts >= min_movie_ratings].index
    
    # Define encoders
    u_enc = LabelEncoder()
    m_enc = LabelEncoder()
    
    # Fit encoders
    u_enc.fit(ratings['userId'])
    m_enc.fit(valid_movie_ids)
    
    # Filter movies dataframe
    movies = movies[movies['movieId'].isin(valid_movie_ids)]
    
    # Load model
    with open('models/ncf_model.pkl', 'rb') as f:
        model = pickle.load(f)
        
    return movies, model, u_enc, m_enc

# Load resources globally
movies, model, u_enc, m_enc = load_resources()

# 3. Streamlit UI
st.title("🎬 Movie Recommender System")

# Interactive Controls
col1, col2 = st.columns(2)
with col1:
    raw_user_id = st.number_input("Enter User ID", min_value=0, max_value=610, value=1)
with col2:
    num_recs = st.slider("Number of recommendations", 1, 10, 5)

# Genre Filtering
all_genres = set('|'.join(movies['genres']).split('|'))
selected_genres = st.multiselect("Filter by Genre", sorted(list(all_genres)))

if st.button("Recommend"):
    try:
        # Check if user exists
        if raw_user_id not in u_enc.classes_:
            st.error("This User ID was not found in the training data.")
        else:
            encoded_user_id = u_enc.transform([raw_user_id])[0]
            
            # Prepare valid movies
            encoded_movie_ids = m_enc.transform(movies['movieId'].unique())
            
            # Prepare Tensors
            user_tensor = torch.LongTensor([encoded_user_id] * len(encoded_movie_ids))
            movie_tensor = torch.LongTensor(encoded_movie_ids)
            
            # Inference
            model.eval()
            with torch.no_grad():
                preds = model(user_tensor, movie_tensor)
            
            # Create results
            results = pd.DataFrame({
                'title': movies['title'].values,
                'genres': movies['genres'].values,
                'predicted_rating': preds.numpy()
            })
            
            # Apply Genre Filter
            if selected_genres:
                results = results[results['genres'].apply(lambda x: any(g in x for g in selected_genres))]
            
            # Sort results
            top_recs = results.sort_values('predicted_rating', ascending=False).head(num_recs)
            
            st.write(f"### Top {num_recs} Recommendations")
            
            # Display as interactive table
            display_df = top_recs[['title', 'predicted_rating']].copy()
            display_df.columns = ['Movie Title', 'Predicted Rating']
            
            st.dataframe(
                display_df, 
                use_container_width=True, 
                hide_index=True
            )
            
    except Exception as e:
        st.error(f"An error occurred: {e}")