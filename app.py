import torch
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
import gradio as gr
from PIL import Image

class GeoGuessr:
    def __init__(self):
        self.model=CLIPModel.from_pretrained("openai/clip-vit-large-patch14")
        self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
        self.texts = [     "Afghanistan", "Albania", "Algeria", "Andorra", "Angola", "Antigua and Barbuda",
                      "Argentina", "Armenia", "Australia", "Austria", "Azerbaijan", "Bahamas", "Bahrain",
                      "Bangladesh", "Barbados", "Belarus", "Belgium", "Belize", "Benin", "Bhutan", "Bolivia",
                      "Bosnia and Herzegovina", "Botswana", "Brazil", "Brunei", "Bulgaria", "Burkina Faso",
                      "Burundi", "Cabo Verde", "Cambodia", "Cameroon", "Canada", "Central African Republic",
                      "Chad", "Chile", "China", "Colombia", "Comoros", "Congo (Congo-Brazzaville)", "Costa Rica",
                      "Croatia", "Cuba", "Cyprus", "Czechia (Czech Republic)", "Denmark", "Djibouti", "Dominica",
                      "Dominican Republic", "Ecuador", "Egypt", "El Salvador", "Equatorial Guinea", "Eritrea",
                      "Estonia", "Eswatini ", "Ethiopia", "Fiji", "Finland", "France", "Gabon",
                      "Gambia", "Georgia", "Germany", "Ghana", "Greece", "Grenada", "Guatemala", "Guinea",
                      "Guinea-Bissau", "Guyana", "Haiti", "Honduras", "Hungary", "Iceland", "India", "Indonesia",
                      "Iran", "Iraq", "Ireland", "Israel", "Italy", "Jamaica", "Japan", "Jordan", "Kazakhstan",
                      "Kenya", "Kiribati", "Kuwait", "Kyrgyzstan", "Laos", "Latvia", "Lebanon", "Lesotho",
                      "Liberia", "Libya", "Liechtenstein", "Lithuania", "Luxembourg", "Madagascar", "Malawi",
                      "Malaysia", "Maldives", "Mali", "Malta", "Marshall Islands", "Mauritania", "Mauritius",
                      "Mexico", "Micronesia", "Moldova", "Monaco", "Mongolia", "Montenegro", "Morocco",
                      "Mozambique", "Myanmar (formerly Burma)", "Namibia", "Nauru", "Nepal", "Netherlands",
                      "New Zealand", "Nicaragua", "Niger", "Nigeria", "North Korea", "North Macedonia",
                      "Norway", "Oman", "Pakistan", "Palau", "Palestine State", "Panama", "Papua New Guinea",
                      "Paraguay", "Peru", "Philippines", "Poland", "Portugal", "Qatar", "Romania", "Russia",
                      "Rwanda", "Saint Kitts and Nevis", "Saint Lucia", "Saint Vincent and the Grenadines",
                      "Samoa", "San Marino", "Sao Tome and Principe", "Saudi Arabia", "Senegal", "Serbia",
                      "Seychelles", "Sierra Leone", "Singapore", "Slovakia", "Slovenia", "Solomon Islands",
                      "Somalia", "South Africa", "South Sudan", "Spain", "Sri Lanka", "Sudan", "Suriname",
                      "Sweden", "Switzerland", "Syria", "Taiwan", "Tajikistan", "Tanzania", "Thailand",
                      "Timor-Leste", "Turkey", "Turkmenistan", "Tuvalu", "Uganda", "Ukraine", "United Arab Emirates",
                      "United Kingdom", "United States of America", "Uruguay", "Uzbekistan", "Vanuatu",
                      "Vatican City", "Venezuela", "Vietnam", "Yemen", "Zambia", "Zimbabwe"
                  ]
    def get_country(self,image_path):
      image = Image.open(image_path)  # Replace with your image path
      inputs = self.processor(text=self.texts, images=image, return_tensors="pt", padding=True)

      # Move tensors to GPU if available
      device = "cuda" if torch.cuda.is_available() else "cpu"
      self.model.to(device)
      inputs = {k: v.to(device) for k, v in inputs.items()}

      # Perform inference
      with torch.no_grad():
          logits_per_image, logits_per_text = self.model(**inputs, return_dict=True).logits_per_image,self.model(**inputs, return_dict=True).logits_per_text

      # Calculate similarity scores
      image_features = logits_per_image.squeeze(0).cpu().numpy()
      text_features = logits_per_text.squeeze(0).cpu().numpy()

      # Find the text with the highest similarity
      similarity_scores = torch.softmax(logits_per_image, dim=1).cpu().numpy()[0]
      country_scores = list(zip(self.texts, similarity_scores))

      # Sort the list of tuples by similarity scores in descending order
      sorted_country_scores = sorted(country_scores, key=lambda x: x[1], reverse=True)

      # Print the top 10 countries with the highest similarity scores
      print("Top 10 Countries Based on Similarity Scores:")
      
      outlist=[]
      for country, score in sorted_country_scores[:10]:
          # print(f"{country}: {score:.4f}")
          outlist.append(dict(zip([country],[str(score)])))
      return str(outlist)


geo=GeoGuessr()
# Define the Gradio interface
def gradio_interface(image):
    # Save the image temporarily and pass its path to the class function
    image_path = "temp_image.png"
    image.save(image_path)
    return geo.get_country(image_path)

# Create the Gradio app
interface = gr.Interface(
    fn=gradio_interface,  # The function to call for predictions
    inputs=gr.Image(type="pil"),  # Input component (expects a PIL Image)
    outputs='text',  # Output component (returns text)
    title="GEoGuessr Helper",  # Optional title
    description="""Upload the image and get top 10 probable places.
                    Can be run on either CPU (currently used) or GPU (for faster inference).
                    """,  # Optional description
)

# Launch the Gradio interface
interface.launch()