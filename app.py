import streamlit as st
from utils.document_processor import parse_uploaded_file
from utils.nlp_models import extract_features, get_embeddings, check_compliance

# Configure Streamlit
st.set_page_config(page_title="Patentability AI Tool", layout="wide")
st.title("🔍 Patentability Assessment Tool")

# File Uploaders
invention_file = st.file_uploader("Upload Invention Description", 
                                 type=["pdf", "txt", "docx"])
prior_art_files = st.file_uploader("Upload Prior Art Documents", 
                                  type=["pdf", "txt", "docx"], 
                                  accept_multiple_files=True)

if invention_file and prior_art_files:
    # Process Files
    invention_text = parse_uploaded_file(invention_file)
    prior_art_texts = [parse_uploaded_file(f) for f in prior_art_files]

    # Feature Extraction
    invention_features = extract_features(invention_text)
    invention_embedding = get_embeddings(invention_text)

    # Prior Art Similarity
    similarities = []
    for art_text in prior_art_texts:
        art_embedding = get_embeddings(art_text)
        similarity = cosine_similarity([invention_embedding], [art_embedding])[0][0]
        similarities.append(similarity * 100)  # Convert to percentage

    # Display Results
    st.subheader("🔬 Analysis Results")
    col1, col2 = st.columns(2)

    with col1:
        st.metric("Average Novelty Score", 
                 f"{100 - sum(similarities)/len(similarities):.1f}%")
        st.write("### Key Technical Features")
        st.write(invention_features[:10])  # Show top 10 features

    with col2:
        st.write("### Top Prior Art Matches")
        for i, sim in enumerate(sorted(similarities, reverse=True)[:3]):
            st.write(f"{i+1}. {sim:.1f}% similar")

    # Compliance Check
    st.subheader("⚠️ Compliance Warnings")
    issues = check_compliance(invention_text)
    for issue in issues:
        st.error(issue)