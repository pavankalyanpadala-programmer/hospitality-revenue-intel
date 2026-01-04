"""
Hotel Booking Cancellation Prediction App

Streamlit application for predicting hotel booking cancellations using a trained model.
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
import sys
import json
import os
import matplotlib.pyplot as plt

# Try to import wordcloud, but handle gracefully if not available
try:
    from wordcloud import WordCloud
    import matplotlib.pyplot as plt
    WORDCLOUD_AVAILABLE = True
except ImportError:
    WORDCLOUD_AVAILABLE = False

# Add parent directory to path to import from src if needed
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


@st.cache_data
def load_data():
    """Load hotel bookings data with caching."""
    processed_path = project_root / 'data' / 'processed' / 'hotel_bookings_clean.csv'
    raw_path = project_root / 'data' / 'raw' / 'hotel_bookings.csv'
    
    if processed_path.exists():
        return pd.read_csv(processed_path)
    elif raw_path.exists():
        return pd.read_csv(raw_path)
    else:
        return None


@st.cache_data
def load_clustered_data():
    """Load hotel bookings data with clusters, with caching and fallback."""
    clustered_path = project_root / 'data' / 'processed' / 'hotel_bookings_with_clusters.csv'
    raw_path = project_root / 'data' / 'raw' / 'hotel_bookings.csv'
    
    if clustered_path.exists():
        return pd.read_csv(clustered_path)
    elif raw_path.exists():
        # Fallback to raw data if clustered data doesn't exist
        return pd.read_csv(raw_path)
    else:
        return None


@st.cache_data
def load_review_data():
    """Load hotel reviews data with sentiment, with caching."""
    review_path = project_root / 'data' / 'processed' / 'hotel_reviews_with_sentiment.csv'
    
    if review_path.exists():
        return pd.read_csv(review_path)
    else:
        return None


@st.cache_data
def load_review_keywords():
    """Load review keywords JSON, with caching."""
    keywords_path = project_root / 'data' / 'processed' / 'review_keywords.json'
    
    if keywords_path.exists():
        try:
            with open(keywords_path, 'r') as f:
                return json.load(f)
        except Exception:
            return None
    else:
        return None


def generate_cluster_description(df_cluster, cluster_num):
    """Generate a description for a cluster based on its characteristics."""
    if df_cluster.empty:
        return f"Cluster {cluster_num}: No data available"
    
    # Calculate key metrics
    avg_adr = df_cluster['adr'].mean() if 'adr' in df_cluster.columns else None
    avg_length_stay = (df_cluster['stays_in_weekend_nights'] + df_cluster['stays_in_week_nights']).mean() if 'stays_in_weekend_nights' in df_cluster.columns else None
    top_market_segment = df_cluster['market_segment'].mode()[0] if 'market_segment' in df_cluster.columns and not df_cluster['market_segment'].mode().empty else None
    avg_lead_time = df_cluster['lead_time'].mean() if 'lead_time' in df_cluster.columns else None
    
    # Build description
    parts = []
    
    # ADR-based description
    if avg_adr is not None:
        if avg_adr < 80:
            parts.append("budget")
        elif avg_adr < 120:
            parts.append("mid-range")
        else:
            parts.append("premium")
    
    # Length of stay
    if avg_length_stay is not None:
        if avg_length_stay < 2:
            parts.append("short-stay")
        elif avg_length_stay < 4:
            parts.append("medium-stay")
        else:
            parts.append("extended-stay")
    
    # Market segment
    if top_market_segment:
        parts.append(f"({top_market_segment})")
    
    description = " ".join(parts) if parts else "mixed characteristics"
    return f"Cluster {cluster_num}: {description.title()} travelers"


def compute_cluster_stats(df_clustered):
    """Compute statistics for each cluster."""
    if df_clustered is None or df_clustered.empty or 'guest_cluster' not in df_clustered.columns:
        return None
    
    cluster_stats = []
    clusters = sorted(df_clustered['guest_cluster'].unique())
    
    for cluster in clusters:
        df_cluster = df_clustered[df_clustered['guest_cluster'] == cluster]
        
        stats = {
            'cluster': cluster,
            'count': len(df_cluster),
            'avg_adr': df_cluster['adr'].mean() if 'adr' in df_cluster.columns else None,
            'avg_length_stay': (df_cluster['stays_in_weekend_nights'] + df_cluster['stays_in_week_nights']).mean() 
                              if 'stays_in_weekend_nights' in df_cluster.columns and 'stays_in_week_nights' in df_cluster.columns else None,
            'description': generate_cluster_description(df_cluster, cluster)
        }
        cluster_stats.append(stats)
    
    return cluster_stats


def compute_kpis(df):
    """Compute key performance indicators from the dataset."""
    if df is None or df.empty:
        return None
    
    total_bookings = len(df)
    
    # Cancellation rate
    if 'is_canceled' in df.columns:
        cancellation_rate = df['is_canceled'].mean() * 100
    else:
        cancellation_rate = None
    
    # Average ADR
    if 'adr' in df.columns:
        avg_adr = df['adr'].mean()
    else:
        avg_adr = None
    
    # Average length of stay
    if 'stays_in_weekend_nights' in df.columns and 'stays_in_week_nights' in df.columns:
        avg_length_of_stay = (df['stays_in_weekend_nights'] + df['stays_in_week_nights']).mean()
    else:
        avg_length_of_stay = None
    
    return {
        'total_bookings': total_bookings,
        'cancellation_rate': cancellation_rate,
        'avg_adr': avg_adr,
        'avg_length_of_stay': avg_length_of_stay
    }


def load_model(model_path):
    """Load the trained model pipeline."""
    try:
        model = joblib.load(model_path)
        return model
    except FileNotFoundError:
        st.error(f"Model file not found at: {model_path}")
        st.error("Please ensure the model has been trained and saved at models/cancellation_model.joblib")
        st.stop()
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        st.stop()


def create_input_dataframe(inputs):
    """Create a DataFrame with user inputs, including default values for missing features."""
    # All features the model expects (excluding is_canceled, reservation_status, reservation_status_date)
    all_features = [
        'hotel', 'lead_time', 'arrival_date_year', 'arrival_date_month',
        'arrival_date_week_number', 'arrival_date_day_of_month',
        'stays_in_weekend_nights', 'stays_in_week_nights', 'adults', 'children', 'babies',
        'meal', 'country', 'market_segment', 'distribution_channel',
        'is_repeated_guest', 'previous_cancellations', 'previous_bookings_not_canceled',
        'reserved_room_type', 'assigned_room_type', 'booking_changes',
        'deposit_type', 'agent', 'company', 'days_in_waiting_list',
        'customer_type', 'adr', 'required_car_parking_spaces', 'total_of_special_requests'
    ]
    
    # Start with the user inputs
    data = inputs.copy()
    
    # Add default values for features not in the form but needed by the model
    # These defaults are based on common values in the dataset
    defaults = {
        'hotel': 'City Hotel',
        'arrival_date_week_number': 1,
        'arrival_date_day_of_month': 1,
        'is_repeated_guest': 0,
        'previous_cancellations': 0,
        'previous_bookings_not_canceled': 0,
        'assigned_room_type': data.get('reserved_room_type', 'A'),
        'booking_changes': 0,
        'agent': np.nan,
        'company': np.nan,
        'days_in_waiting_list': 0,
        'required_car_parking_spaces': 0,
        'total_of_special_requests': 0,
        'country': 'PRT',
    }
    
    # Merge defaults (only if key not already in data)
    for key, value in defaults.items():
        if key not in data:
            data[key] = value
    
    # Create DataFrame ensuring all required columns are present
    df_dict = {}
    for feature in all_features:
        if feature in data:
            df_dict[feature] = [data[feature]]
        else:
            # Fallback default (shouldn't happen, but just in case)
            if feature in ['agent', 'company']:
                df_dict[feature] = [np.nan]
            elif feature in defaults:
                df_dict[feature] = [defaults[feature]]
            else:
                # This shouldn't happen, but provide a safe default
                df_dict[feature] = [0] if feature not in ['hotel', 'arrival_date_month', 'meal', 
                                                          'country', 'market_segment', 'distribution_channel',
                                                          'reserved_room_type', 'assigned_room_type',
                                                          'deposit_type', 'customer_type'] else ['']
    
    df = pd.DataFrame(df_dict)
    
    # Ensure correct data types for numeric columns
    numeric_cols = ['lead_time', 'arrival_date_year', 'arrival_date_week_number',
                   'arrival_date_day_of_month', 'stays_in_weekend_nights', 'stays_in_week_nights',
                   'adults', 'children', 'babies', 'is_repeated_guest', 'previous_cancellations',
                   'previous_bookings_not_canceled', 'booking_changes', 'days_in_waiting_list',
                   'adr', 'required_car_parking_spaces', 'total_of_special_requests', 'agent', 'company']
    
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    return df


def main():
    """Main Streamlit app."""
    st.set_page_config(
        page_title="Hotel Booking Cancellation Predictor",
        page_icon="🏨",
        layout="wide"
    )
    
    st.title("🏨 Hotel Booking Cancellation Predictor")
    
    # Load data for KPIs and visualizations
    df = load_data()
    
    if df is not None:
        # Compute KPIs
        kpis = compute_kpis(df)
        
        if kpis:
            # Display KPI metrics
            st.header("📈 Key Performance Indicators")
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric(
                    label="Total Bookings",
                    value=f"{kpis['total_bookings']:,}"
                )
            
            with col2:
                if kpis['cancellation_rate'] is not None:
                    st.metric(
                        label="Cancellation Rate",
                        value=f"{kpis['cancellation_rate']:.1f}%"
                    )
                else:
                    st.metric(label="Cancellation Rate", value="N/A")
            
            with col3:
                if kpis['avg_adr'] is not None:
                    st.metric(
                        label="Average ADR",
                        value=f"€{kpis['avg_adr']:.2f}"
                    )
                else:
                    st.metric(label="Average ADR", value="N/A")
            
            with col4:
                if kpis['avg_length_of_stay'] is not None:
                    st.metric(
                        label="Avg Length of Stay",
                        value=f"{kpis['avg_length_of_stay']:.1f} nights"
                    )
                else:
                    st.metric(label="Avg Length of Stay", value="N/A")
            
            st.divider()
            
            # Visualizations
            st.header("📊 Data Visualizations")
            
            viz_col1, viz_col2 = st.columns(2)
            
            with viz_col1:
                st.subheader("Bookings by Arrival Month and Year")
                if 'arrival_date_month' in df.columns and 'arrival_date_year' in df.columns:
                    # Map month names to numbers for proper sorting
                    month_order = {
                        'January': 1, 'February': 2, 'March': 3, 'April': 4,
                        'May': 5, 'June': 6, 'July': 7, 'August': 8,
                        'September': 9, 'October': 10, 'November': 11, 'December': 12
                    }
                    df_viz = df.copy()
                    df_viz['month_num'] = df_viz['arrival_date_month'].map(month_order)
                    # Group by year and month, then count
                    bookings_by_month = df_viz.groupby(['arrival_date_year', 'arrival_date_month', 'month_num']).size().reset_index(name='count')
                    bookings_by_month = bookings_by_month.sort_values(['arrival_date_year', 'month_num'])
                    # Create a readable label for the x-axis
                    bookings_by_month['month_year_label'] = bookings_by_month['arrival_date_month'] + ' ' + bookings_by_month['arrival_date_year'].astype(str)
                    # Use line chart for time series
                    st.line_chart(bookings_by_month.set_index('month_year_label')['count'])
                else:
                    st.info("Arrival date columns not available in dataset")
            
            with viz_col2:
                st.subheader("Cancellation Rate by Market Segment")
                if 'market_segment' in df.columns and 'is_canceled' in df.columns:
                    cancellation_by_segment = df.groupby('market_segment')['is_canceled'].agg(['mean', 'count']).reset_index()
                    cancellation_by_segment.columns = ['market_segment', 'cancellation_rate', 'count']
                    cancellation_by_segment = cancellation_by_segment.sort_values('cancellation_rate', ascending=False)
                    cancellation_by_segment['cancellation_rate_pct'] = cancellation_by_segment['cancellation_rate'] * 100
                    st.bar_chart(cancellation_by_segment.set_index('market_segment')['cancellation_rate_pct'])
                else:
                    st.info("Market segment or cancellation data not available")
            
            st.divider()
        else:
            st.warning("⚠️ Could not compute KPIs from the dataset")
    else:
        st.warning("⚠️ Data file not found. Please ensure data/raw/hotel_bookings.csv exists.")
        st.info("KPIs and visualizations will not be available, but predictions can still be made if the model is available.")
    
    # Guest Segments Section (in expander to avoid cluttering)
    df_clustered = load_clustered_data()
    
    if df_clustered is not None and 'guest_cluster' in df_clustered.columns:
        with st.expander("👥 Guest Segments Explorer", expanded=False):
            st.header("👥 Guest Segments")
            
            # Use tabs to organize the segment explorer
            segment_tab1, segment_tab2 = st.tabs(["📊 Overview", "📈 Detailed Analysis"])
            
            with segment_tab1:
                # Compute cluster statistics
                cluster_stats = compute_cluster_stats(df_clustered)
                
                if cluster_stats:
                    st.subheader("Cluster Summary")
                    
                    # Display cluster counts in metric cards
                    n_clusters = len(cluster_stats)
                    cols = st.columns(n_clusters)
                    
                    for i, stats in enumerate(cluster_stats):
                        with cols[i]:
                            st.metric(
                                label=f"Cluster {stats['cluster']}",
                                value=f"{stats['count']:,}",
                                help=stats['description']
                            )
                    
                    st.markdown("---")
                    
                    # Display cluster descriptions
                    st.subheader("Cluster Descriptions")
                    for stats in cluster_stats:
                        st.markdown(f"**{stats['description']}**")
            
            with segment_tab2:
                st.subheader("Visualizations")
                
                # Visualization 1: Average ADR by cluster
                viz1_col1, viz1_col2 = st.columns(2)
                
                with viz1_col1:
                    st.subheader("Average ADR by Cluster")
                    if 'adr' in df_clustered.columns:
                        adr_by_cluster = df_clustered.groupby('guest_cluster')['adr'].mean().sort_index()
                        adr_by_cluster.index = [f"Cluster {i}" for i in adr_by_cluster.index]
                        st.bar_chart(adr_by_cluster)
                    else:
                        st.info("ADR data not available")
                
                with viz1_col2:
                    st.subheader("Average Length of Stay by Cluster")
                    if 'stays_in_weekend_nights' in df_clustered.columns and 'stays_in_week_nights' in df_clustered.columns:
                        df_clustered_viz = df_clustered.copy()
                        df_clustered_viz['total_nights'] = df_clustered_viz['stays_in_weekend_nights'] + df_clustered_viz['stays_in_week_nights']
                        length_stay_by_cluster = df_clustered_viz.groupby('guest_cluster')['total_nights'].mean().sort_index()
                        length_stay_by_cluster.index = [f"Cluster {i}" for i in length_stay_by_cluster.index]
                        st.bar_chart(length_stay_by_cluster)
                    else:
                        st.info("Stay duration data not available")
                
                st.markdown("---")
                
                # Visualization 3: Market segment distribution by cluster
                st.subheader("Market Segment Distribution by Cluster")
                if 'market_segment' in df_clustered.columns:
                    # Create a pivot table for market segment by cluster
                    segment_dist = df_clustered.groupby(['guest_cluster', 'market_segment']).size().reset_index(name='count')
                    segment_pivot = segment_dist.pivot(index='market_segment', columns='guest_cluster', values='count').fillna(0)
                    # Rename columns
                    segment_pivot.columns = [f"Cluster {col}" for col in segment_pivot.columns]
                    st.bar_chart(segment_pivot)
                else:
                    st.info("Market segment data not available")
    
    elif df_clustered is not None:
        # Data loaded but no cluster column
        with st.expander("👥 Guest Segments", expanded=False):
            st.info("Clustered data file found, but 'guest_cluster' column is missing. Please ensure the data contains cluster assignments.")
    else:
        # No clustered data available - don't show expander if data doesn't exist
        pass
    
    st.divider()
    
    # Review Insights Section
    df_reviews = load_review_data()
    review_keywords = load_review_keywords()
    
    if df_reviews is not None and 'sentiment_label' in df_reviews.columns:
        with st.expander("💬 Review Insights", expanded=False):
            st.header("💬 Review Insights")
            
            # Compute sentiment statistics (shared across tabs)
            total_reviews = len(df_reviews)
            sentiment_counts = df_reviews['sentiment_label'].value_counts()
            sentiment_pct = df_reviews['sentiment_label'].value_counts(normalize=True) * 100
            
            # Use tabs to organize the review insights
            review_tab1, review_tab2, review_tab3 = st.tabs(["📊 Summary", "📈 Visualizations", "📝 Sample Reviews"])
            
            with review_tab1:
                # Summary metrics
                st.subheader("Summary Metrics")
                
                # Display metrics in columns
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("Total Reviews", f"{total_reviews:,}")
                
                with col2:
                    positive_count = sentiment_counts.get('positive', 0)
                    positive_pct = sentiment_pct.get('positive', 0)
                    st.metric("Positive Reviews", f"{positive_count:,}", f"{positive_pct:.1f}%")
                
                with col3:
                    negative_count = sentiment_counts.get('negative', 0)
                    negative_pct = sentiment_pct.get('negative', 0)
                    st.metric("Negative Reviews", f"{negative_count:,}", f"{negative_pct:.1f}%")
                
                with col4:
                    neutral_count = sentiment_counts.get('neutral', 0)
                    neutral_pct = sentiment_pct.get('neutral', 0)
                    st.metric("Neutral Reviews", f"{neutral_count:,}", f"{neutral_pct:.1f}%")
                
                st.markdown("---")
                
                # Sentiment distribution table
                st.subheader("Sentiment Distribution")
                dist_df = pd.DataFrame({
                    'Sentiment': sentiment_counts.index,
                    'Count': sentiment_counts.values,
                    'Percentage': [f"{sentiment_pct[sent]:.2f}%" for sent in sentiment_counts.index]
                })
                st.dataframe(dist_df, use_container_width=True, hide_index=True)
            
            with review_tab2:
                st.subheader("Visualizations")
                
                # Sentiment distribution chart
                viz_col1, viz_col2 = st.columns(2)
                
                with viz_col1:
                    st.subheader("Sentiment Distribution (Bar Chart)")
                    st.bar_chart(sentiment_counts)
                
                with viz_col2:
                    st.subheader("Sentiment Distribution (Counts)")
                    # Display sentiment counts as a table for clarity
                    sentiment_display = pd.DataFrame({
                        'Sentiment': sentiment_counts.index,
                        'Count': sentiment_counts.values
                    })
                    st.dataframe(sentiment_display, use_container_width=True, hide_index=True)
                
                st.markdown("---")
                
                # Keywords visualization
                if review_keywords is not None:
                    st.subheader("Top Keywords by Sentiment")
                    
                    # Get top keywords (assuming lists are in the JSON)
                    positive_keywords = review_keywords.get('positive', [])
                    negative_keywords = review_keywords.get('negative', [])
                    
                    # Handle if keywords are strings (JSON serialized lists)
                    if isinstance(positive_keywords, str):
                        import ast
                        positive_keywords = ast.literal_eval(positive_keywords)
                    if isinstance(negative_keywords, str):
                        import ast
                        negative_keywords = ast.literal_eval(negative_keywords)
                    
                    # Take top 15 keywords
                    top_n = 15
                    positive_top = positive_keywords[:top_n] if isinstance(positive_keywords, list) else []
                    negative_top = negative_keywords[:top_n] if isinstance(negative_keywords, list) else []
                    
                    keyword_col1, keyword_col2 = st.columns(2)
                    
                    with keyword_col1:
                        st.subheader(f"Top {len(positive_top)} Positive Keywords")
                        if positive_top:
                            # Create a bar chart using rank order (keywords are already ordered by importance)
                            # Assign weights based on position (higher rank = higher weight)
                            weights = [len(positive_top) - i for i in range(len(positive_top))]
                            pos_keyword_df = pd.DataFrame({
                                'Keyword': positive_top,
                                'Importance': weights
                            })
                            # Reverse for better visualization (highest importance on top)
                            pos_keyword_df = pos_keyword_df.iloc[::-1]
                            st.bar_chart(pos_keyword_df.set_index('Keyword')['Importance'])
                        else:
                            st.info("No positive keywords available")
                    
                    with keyword_col2:
                        st.subheader(f"Top {len(negative_top)} Negative Keywords")
                        if negative_top:
                            weights = [len(negative_top) - i for i in range(len(negative_top))]
                            neg_keyword_df = pd.DataFrame({
                                'Keyword': negative_top,
                                'Importance': weights
                            })
                            neg_keyword_df = neg_keyword_df.iloc[::-1]
                            st.bar_chart(neg_keyword_df.set_index('Keyword')['Importance'])
                        else:
                            st.info("No negative keywords available")
                    
                    # Word clouds (optional)
                    if WORDCLOUD_AVAILABLE:
                        st.markdown("---")
                        st.subheader("Keyword Word Clouds")
                        
                        wordcloud_col1, wordcloud_col2 = st.columns(2)
                        
                        with wordcloud_col1:
                            st.markdown("**Positive Keywords**")
                            if positive_top:
                                # Create word cloud
                                wordcloud_text = ' '.join(positive_top)
                                wordcloud = WordCloud(width=400, height=300, background_color='white').generate(wordcloud_text)
                                fig, ax = plt.subplots(figsize=(8, 6))
                                ax.imshow(wordcloud, interpolation='bilinear')
                                ax.axis('off')
                                st.pyplot(fig)
                                plt.close(fig)
                        
                        with wordcloud_col2:
                            st.markdown("**Negative Keywords**")
                            if negative_top:
                                wordcloud_text = ' '.join(negative_top)
                                wordcloud = WordCloud(width=400, height=300, background_color='white').generate(wordcloud_text)
                                fig, ax = plt.subplots(figsize=(8, 6))
                                ax.imshow(wordcloud, interpolation='bilinear')
                                ax.axis('off')
                                st.pyplot(fig)
                                plt.close(fig)
                else:
                    st.info("Keyword data not available. Please ensure data/processed/review_keywords.json exists.")
            
            with review_tab3:
                st.subheader("Sample Reviews")
                
                # Get sample positive and negative reviews
                if 'sentiment_compound' in df_reviews.columns and 'Combined_Review' in df_reviews.columns:
                    # Positive reviews (highest sentiment scores)
                    positive_reviews = df_reviews[df_reviews['sentiment_label'] == 'positive'].nlargest(3, 'sentiment_compound')
                    
                    st.markdown("### 🌟 Top Positive Reviews")
                    for idx, (_, review) in enumerate(positive_reviews.iterrows(), 1):
                        with st.expander(f"Positive Review #{idx} (Score: {review['sentiment_compound']:.3f})", expanded=False):
                            st.write(review['Combined_Review'][:500] + "..." if len(review['Combined_Review']) > 500 else review['Combined_Review'])
                    
                    st.markdown("---")
                    
                    # Negative reviews (lowest sentiment scores)
                    negative_reviews = df_reviews[df_reviews['sentiment_label'] == 'negative'].nsmallest(3, 'sentiment_compound')
                    
                    st.markdown("### ⚠️ Most Negative Reviews")
                    for idx, (_, review) in enumerate(negative_reviews.iterrows(), 1):
                        with st.expander(f"Negative Review #{idx} (Score: {review['sentiment_compound']:.3f})", expanded=False):
                            st.write(review['Combined_Review'][:500] + "..." if len(review['Combined_Review']) > 500 else review['Combined_Review'])
                else:
                    st.info("Review text or sentiment scores not available in the dataset.")
    elif df_reviews is not None:
        # Data loaded but missing sentiment_label column
        with st.expander("💬 Review Insights", expanded=False):
            st.info("Review data file found, but 'sentiment_label' column is missing. Please ensure the data contains sentiment labels.")
    else:
        # No review data available - don't show expander if data doesn't exist
        pass
    
    st.divider()
    
    st.header("🔮 Cancellation Prediction")
    st.markdown("Enter booking details in the sidebar to predict the probability of cancellation.")
    
    # Load model
    model_path = project_root / 'models' / 'cancellation_model.joblib'
    
    # Check if model exists before loading
    if not model_path.exists():
        st.error(f"❌ Model file not found at: {model_path}")
        st.error("Please ensure the model has been trained first by running:")
        st.code("python src/models/cancellation_model.py", language="bash")
        st.stop()
    
    model = load_model(model_path)
    
    # Sidebar for input form
    with st.sidebar:
        st.header("📝 Booking Details")
        
        # Numeric inputs
        lead_time = st.number_input("Lead Time (days)", min_value=0, max_value=800, value=100, step=1,
                                   help="Number of days between booking and arrival")
        
        arrival_date_year = st.selectbox("Arrival Year", [2015, 2016, 2017], index=2)
        
        arrival_date_month = st.selectbox(
            "Arrival Month",
            ["January", "February", "March", "April", "May", "June",
             "July", "August", "September", "October", "November", "December"],
            index=6
        )
        
        stays_in_weekend_nights = st.number_input("Weekend Nights", min_value=0, max_value=20, value=0, step=1)
        
        stays_in_week_nights = st.number_input("Week Nights", min_value=0, max_value=50, value=2, step=1)
        
        adults = st.number_input("Adults", min_value=0, max_value=50, value=2, step=1)
        
        children = st.number_input("Children", min_value=0.0, max_value=10.0, value=0.0, step=1.0)
        
        babies = st.number_input("Babies", min_value=0, max_value=10, value=0, step=1)
        
        adr = st.number_input("ADR (Average Daily Rate)", min_value=0.0, max_value=5500.0, value=100.0, step=10.0,
                             help="Average daily rate in EUR")
        
        # Categorical inputs
        meal = st.selectbox(
            "Meal Type",
            ["BB", "FB", "HB", "SC", "Undefined"],
            index=0,
            help="BB: Bed & Breakfast, FB: Full Board, HB: Half Board, SC: Self Catering"
        )
        
        market_segment = st.selectbox(
            "Market Segment",
            ["Direct", "Corporate", "Online TA", "Offline TA/TO", "Complementary", "Groups", "Undefined", "Aviation"],
            index=2
        )
        
        distribution_channel = st.selectbox(
            "Distribution Channel",
            ["Direct", "Corporate", "TA/TO", "Undefined", "GDS"],
            index=2
        )
        
        reserved_room_type = st.selectbox(
            "Reserved Room Type",
            ["A", "B", "C", "D", "E", "F", "G", "H", "L", "P"],
            index=0
        )
        
        customer_type = st.selectbox(
            "Customer Type",
            ["Transient", "Contract", "Transient-Party", "Group"],
            index=0
        )
        
        deposit_type = st.selectbox(
            "Deposit Type",
            ["No Deposit", "Refundable", "Non Refund"],
            index=0
        )
        
        # Prediction button
        predict_button = st.button("🔮 Predict Cancellation Risk", type="primary", use_container_width=True)
    
    # Main content area
    if predict_button:
        # Create input dictionary
        input_data = {
            'lead_time': lead_time,
            'arrival_date_year': arrival_date_year,
            'arrival_date_month': arrival_date_month,
            'stays_in_weekend_nights': stays_in_weekend_nights,
            'stays_in_week_nights': stays_in_week_nights,
            'adults': adults,
            'children': children,
            'babies': babies,
            'meal': meal,
            'market_segment': market_segment,
            'distribution_channel': distribution_channel,
            'reserved_room_type': reserved_room_type,
            'customer_type': customer_type,
            'adr': adr,
            'deposit_type': deposit_type,
        }
        
        # Create DataFrame
        try:
            input_df = create_input_dataframe(input_data)
            
            # Make prediction
            with st.spinner("Processing prediction..."):
                cancellation_prob = model.predict_proba(input_df)[0, 1]
            
            # Display results
            st.header("📊 Prediction Results")
            
            # Probability display
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.subheader("Cancellation Probability")
                st.metric(
                    label="Probability",
                    value=f"{cancellation_prob:.1%}",
                    delta=f"{(cancellation_prob - 0.5) * 100:.1f}%"
                )
            
            with col2:
                # Risk level
                threshold = 0.5
                if cancellation_prob >= threshold:
                    risk_level = "High Risk ⚠️"
                    risk_color = "red"
                else:
                    risk_level = "Low Risk ✅"
                    risk_color = "green"
                
                st.subheader("Risk Level")
                st.markdown(f"<h3 style='color: {risk_color};'>{risk_level}</h3>", unsafe_allow_html=True)
            
            # Progress bar
            st.progress(cancellation_prob)
            
            # Risk message
            if cancellation_prob >= threshold:
                st.warning(f"⚠️ **High risk of cancellation** ({cancellation_prob:.1%})")
                st.info("Consider implementing retention strategies such as personalized communication, flexible cancellation policies, or special offers.")
            else:
                st.success(f"✅ **Low risk of cancellation** ({cancellation_prob:.1%})")
                st.info("This booking has a lower likelihood of cancellation. Continue with standard booking procedures.")
            
            # Additional information
            with st.expander("📋 Input Summary", expanded=False):
                st.json(input_data)
        
        except Exception as e:
            st.error(f"❌ Error making prediction: {str(e)}")
            st.exception(e)
    
    else:
        # Show instructions when button not clicked
        st.info("👈 Please fill in the booking details in the sidebar and click 'Predict Cancellation Risk' to see the prediction.")
        
        # Show some example information
        with st.expander("ℹ️ About This Predictor", expanded=False):
            st.markdown("""
            This application uses a trained machine learning model to predict the probability 
            that a hotel booking will be cancelled.
            
            **How to use:**
            1. Review the KPIs and visualizations above to understand the booking patterns
            2. Fill in the booking details in the sidebar
            3. Click the "Predict Cancellation Risk" button
            4. Review the predicted probability and risk level
            
            **Risk Threshold:** Predictions above 50% are considered high risk.
            
            **Model:** Random Forest Classifier trained on historical hotel booking data.
            """)


if __name__ == "__main__":
    main()

# ============================================================
# FORECAST TAB
# ============================================================

def load_forecast_data():
    """Load forecasted booking and revenue data."""
    forecast_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'processed', 'booking_forecast.csv')
    
    if not os.path.exists(forecast_path):
        return None
    
    df = pd.read_csv(forecast_path)
    df['date'] = pd.to_datetime(df['date'])
    return df

def show_forecast_tab():
    """Display demand forecast visualizations."""
    st.header("📈 Demand Forecast")
    
    forecast_data = load_forecast_data()
    
    if forecast_data is None:
        st.warning("⚠️ Forecast data not found. Please run the forecasting model first: `python -m src.models.demand_forecasting`")
        return
    
    # Split actual vs forecast
    actual_data = forecast_data[forecast_data['type'] == 'actual'].copy()
    forecast_future = forecast_data[forecast_data['type'] == 'forecast'].copy()
    
    st.write(f"**Historical data:** {len(actual_data)} days")
    st.write(f"**Forecast period:** {len(forecast_future)} days (next {len(forecast_future)} days)")
    
    # Show metrics for last 30 days vs forecast
    last_30_actual = actual_data.tail(30)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        avg_bookings_historical = last_30_actual['bookings'].mean()
        avg_bookings_forecast = forecast_future['bookings'].mean()
        delta_bookings = ((avg_bookings_forecast - avg_bookings_historical) / avg_bookings_historical * 100) if avg_bookings_historical > 0 else 0
        st.metric(
            "Avg Daily Bookings (Forecast)", 
            f"{avg_bookings_forecast:.0f}",
            f"{delta_bookings:+.1f}% vs last 30d"
        )
    
    with col2:
        avg_revenue_historical = last_30_actual['revenue'].mean()
        avg_revenue_forecast = forecast_future['revenue'].mean()
        delta_revenue = ((avg_revenue_forecast - avg_revenue_historical) / avg_revenue_historical * 100) if avg_revenue_historical > 0 else 0
        st.metric(
            "Avg Daily Revenue (Forecast)", 
            f"${avg_revenue_forecast:,.0f}",
            f"{delta_revenue:+.1f}% vs last 30d"
        )
    
    with col3:
        total_forecast_revenue = forecast_future['revenue'].sum()
        st.metric(
            f"Total Revenue ({len(forecast_future)}d forecast)", 
            f"${total_forecast_revenue:,.0f}"
        )
    
    st.markdown("---")
    
    # Bookings forecast chart
    st.subheader("📊 Bookings Forecast")
    
    # Combine last 60 days of actual + forecast
    recent_actual = actual_data.tail(60)
    
    fig_bookings, ax = plt.subplots(figsize=(12, 5))
    ax.plot(recent_actual['date'], recent_actual['bookings'], label='Historical', color='steelblue', linewidth=2)
    ax.plot(forecast_future['date'], forecast_future['bookings'], label='Forecast', color='orange', linewidth=2, linestyle='--')
    ax.axvline(x=actual_data['date'].max(), color='red', linestyle=':', linewidth=1, label='Forecast Start')
    ax.set_xlabel('Date')
    ax.set_ylabel('Daily Bookings')
    ax.set_title('Daily Bookings: Historical vs Forecast')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    plt.tight_layout()
    st.pyplot(fig_bookings)
    
    st.markdown("---")
    
    # Revenue forecast chart
    st.subheader("💰 Revenue Forecast")
    
    fig_revenue, ax = plt.subplots(figsize=(12, 5))
    ax.plot(recent_actual['date'], recent_actual['revenue'], label='Historical', color='green', linewidth=2)
    ax.plot(forecast_future['date'], forecast_future['revenue'], label='Forecast', color='red', linewidth=2, linestyle='--')
    ax.axvline(x=actual_data['date'].max(), color='red', linestyle=':', linewidth=1, label='Forecast Start')
    ax.set_xlabel('Date')
    ax.set_ylabel('Daily Revenue ($)')
    ax.set_title('Daily Revenue: Historical vs Forecast')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    plt.tight_layout()
    st.pyplot(fig_revenue)
    
    # Show forecast table
    with st.expander("📋 View Forecast Data Table"):
        st.dataframe(forecast_future[['date', 'bookings', 'revenue']].head(30))
# ============================================================
# DEMAND FORECAST SECTION
# ============================================================
st.markdown("---")
st.markdown("---")
st.header("📈 Demand Forecast")
show_forecast_tab()
