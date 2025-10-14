import streamlit as st
from MetaFinder2 import VenueRecommender # Import your class from the other file

# --- Page Configuration ---
st.set_page_config(
    page_title="MetaFind",
    page_icon="📚",
    layout="centered"
)

# --- Caching the Recommender ---
@st.cache_resource # This is the key to making the app fast!
def get_recommender():
    """Creates and caches the VenueRecommender instance."""
    return VenueRecommender()

# --- UI Elements ---
st.title("📚 MetaFind: Journal Recommender")
st.write("Enter your manuscript's abstract or keywords below to get journal recommendations from two different methods: a vector search and a direct LLM agent.")

# Get the cached recommender instance
recommender = get_recommender()

# Use a form to group the input and button
with st.form("recommendation_form"):
    user_abstract = st.text_area(
        "Paste your abstract or keyword here:",
        height=250,
        placeholder="e.g., This paper introduces a novel deep learning model..."
    )
    submitted = st.form_submit_button("Get Recommendations")

# --- Backend Logic ---
if submitted and user_abstract:
    with st.spinner("Searching for recommendations... this may take a moment."):
        # Get results from both methods
        # These methods now RETURN strings, ready for display
        index_results = recommender.recommend_from_index(user_abstract)
        llm_results = recommender.recommend_with_llm(user_abstract)
        
        st.divider()

        # Display the results in two columns
        col1, col2 = st.columns(2)
        with col1:
            st.markdown(index_results)
        with col2:
            st.markdown(llm_results)

elif submitted and not user_abstract:
    st.warning("Please enter an abstract or keyword.")
