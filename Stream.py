import streamlit as st
from PIL import Image
from modules import detect_Trailer, Crop_Trailer, extract_top_colors_from_pil_image  # Import your module with processing functions

# Function to process images and generate reports
def process_images(image1, image2):
    # Detect trailers in both images
    annotated_image1, detections1 = detect_Trailer(image1)
    annotated_image2, detections2 = detect_Trailer(image2)
    
    # Crop the trailers
    cropped_image1, brand_name1 = Crop_Trailer(image1, detections1)
    cropped_image2,brand_name2 = Crop_Trailer(image2, detections2)
    
    # Extract color names from the cropped images
    colors1 = extract_top_colors_from_pil_image(cropped_image1, num_colors=2)
    colors2 = extract_top_colors_from_pil_image(cropped_image2, num_colors=2)
    
    # Generate reports
    report1 = {
        'color': ', '.join(colors1),
        'brand': ''.join(brand_name1),  # Replace with actual brand extraction logic
    }
    
    report2 = {
        'color': ', '.join(colors2),
        'brand': ''.join(brand_name2),  # Replace with actual brand extraction logic
    }
    
    # Check for fraud based on color and brand information
    fraud_detected = report1['color'] != report2['color'] or report1['brand'] != report2['brand']
    
    return report1, report2, fraud_detected



def main():
    # Sidebar with buttons
    st.sidebar.title("Navigation")
    option = st.sidebar.radio("Choose an option", ("Upload and Process Images", "View Reports"))

    if option == "Upload and Process Images":
        st.title("Image Upload and Processing")
        
        # Image upload
        col1, col2 = st.columns(2)
        
        with col1:
            image1 = st.file_uploader("Upload Entry Image ", type=['jpg', 'png'])
            if image1:
                st.image(Image.open(image1), caption="Uploaded Image 1", use_column_width=True)
        
        with col2:
            image2 = st.file_uploader("Upload Exit Image", type=['jpg', 'png'])
            if image2:
                st.image(Image.open(image2), caption="Uploaded Image 2", use_column_width=True)
        

        # Process images
        if image1 and image2:
            if st.button("Process"):
                with st.spinner("Processing"):
                    st.session_state['processed'] = True
                    # Process images and generate reports
                    report1, report2, fraud_detected = process_images(Image.open(image1), Image.open(image2))
                    
                    st.session_state['report1'] = report1
                    st.session_state['report2'] = report2
                    st.session_state['fraud_detected'] = fraud_detected
                    
                st.success("Processing complete. Click on 'View Reports' to see the results.")

    elif option == "View Reports":
        if 'report1' in st.session_state and 'report2' in st.session_state:
            st.title("Image Reports")
            
            # Display reports
            st.subheader("Report for Entry Image ")
            st.write(f"Color: {st.session_state['report1']['color']}")
            st.write(f"Brand: {st.session_state['report1']['brand']}")
            
            st.subheader("Report for Exit Image ")
            st.write(f"Color: {st.session_state['report2']['color']}")
            st.write(f"Brand: {st.session_state['report2']['brand']}")
            
            # Fraud detection result
            if st.session_state['fraud_detected']:
                st.markdown('<p style="color: red;">Fraud Detected</p>', unsafe_allow_html=True)
            else:
                st.markdown('<p style="color: green;">No Fraud Detected</p>', unsafe_allow_html=True)
            
        else:
            st.warning("No reports available. Please upload and process images first.")
if __name__ == '__main__':
    main()