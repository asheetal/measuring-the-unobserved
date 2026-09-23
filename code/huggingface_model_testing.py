#Importing Required Classes and Functions
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
from collections import Counter


#Loading the Pre-trained Tokenizer
tokenizer = AutoTokenizer.from_pretrained("pparasurama/raceBERT-ethnicity")

#Loading the Pre-trained Model
model = AutoModelForSequenceClassification.from_pretrained("pparasurama/raceBERT-ethnicity")
model.config.id2label

# Creating the Communication Style Classification Pipeline
pipe = pipeline("text-classification", model=model, tokenizer=tokenizer)

# Using the Pipeline to Classify Text
# detail can be found: https://huggingface.co/docs/transformers/main_classes/pipelines#transformers.pipeline
# pipe("input the text here"). Example as below:

print(pipe("Malik Bendjelloul", return_all_scores=False))

black_french_people = [
    # Political Activists
    "Frantz Fanon",
    "Louis-Georges Tin",
    "Rokhaya Diallo",
    "Sibeth Ndiaye",
    "Susanna Ounei",
    "Stéphane Pocrain",
    "Fodé Sylla",

    # Literature
    "Calixthe Beyala",
    "Aimé Césaire",
    "Suzanne Césaire",
    "Maryse Condé",
    "Raphaël Confiant",
    "Léon Damas",
    "Gerty Dambury",
    "Fatou Diome",
    "David Diop",
    "Édouard Glissant",
    "Viktor Lazlo",
    "René Maran",
    "Daniel Maximin",
    "Jeanne Nardal",
    "Paulette Nardal",
    "Marie NDiaye",
    "Gaël Octavia",
    "Daniel Picouly",
    "Gisèle Pineau",
    "Claude Ribbe",
    "Raphaël Tardon",
    "Guy Tirolien",
    "Joseph Zobel",

    # European / African (or Afro-Caribbean) Descent
    "Alexandre Dumas",
    "Alexandre Dumas fils",
    "Thomas-Alexandre Dumas",
    "Thierry Dusautoir",
    "Chevalier de Saint-Georges",
    "Rudy Gobert",
    "Noémie Lenoir",
    "Chevalier de Meude-Monpas",
    "Chloé Mortaud",
    "Anais Mali",
    "Sonia Rolland",
    "Jo-Wilfried Tsonga",
    "Gaël Monfils",
    "Flora Coquerel",
    "Alicia Aylies",
    "Willy William",
    "Cindy Bruna",
    "Ciryl Gane",

    # Afro-French Members of the French Parliament or Government from Overseas France
    "Jean-Baptiste Belley",
    "Hégésippe Légitimus",
    "Gratien Candace",
    "Blaise Diagne",
    "Ngalandou Diouf",
    "Achille René-Boisneuf",
    "Maurice Satineau",
    "Roger Bambuck",
    "Aimé Césaire",
    "Jean-Louis d'Anglebermes",
    "Félix Éboué",
    "Laura Flessel-Colovic",
    "Serge Letchimy",
    "Gaston Monnerville",
    "Maurice Ponga",
    "Christiane Taubira",
    "Manuéla Kéclard-Mondésir",

    # Afro-French People Elected in Metropolitan France
    "Louis Guizot",
    "Severiano de Heredia",
    "Raphaël Élizé",
    "Élie Bloncourt",
    "Ernest Chénière",
    "Hélène Geoffroy",
    "Maxette Grisoni-Pirbakas",
    "George Pau-Langevin",
    "Arthur Richards",
    "Rama Yade",
    "Harlem Désir",
    "Kofi Yamgnane",
    "Hervé Berville",
    "Seybah Dagoma",
    "Laetitia Avia",
    "Danièle Obono",
    "Nadège Abomangoli",
    "Pap Ndiaye",
    "Rachel Keke",
    "Fanta Berete",
    "Carlos Martens Bilongo",

    # Sports - Basketball
    "Tariq Abdul-Wahad",
    "Alexis Ajinça",
    "Andrew Albicy",
    "Joël Ayayi",
    "Nicolas Batum",
    "Rodrigue Beaubois",
    "Juhann Begarin",
    "Isaïa Cordinier",
    "Boris Diaw",
    "Moustapha Fall",
    "Rudy Gobert",
    "Sandrine Gruda",
    "Mathias Lessort",
    "Timothé Luwawu-Cabarrot",
    "Ian Mahinmi",
    "Amath M'Baye",
    "Joakim Noah",
    "Endéné Miyem",
    "Frank Ntilikina",
    "Tony Parker",
    "Johan Petro",
    "Mickaël Piétrus",
    "Yves Pons",
    "Iliana Rupert",
    "Olivier Sarr",
    "Kevin Séraphin",
    "Diandra Tchatchouang",
    "Axel Toupane",
    "Ronny Turiaf",
    "Valériane Vukosavljević",
    "Victor Wembanyama",
    "Gabby Williams",
    "Guerschon Yabusele",
    "Isabelle Yacoubou",

    # Sports - Rugby
    "Pierre-Henri Azagoh",
    "Demba Bamba",
    "Mathieu Bastareaud",
    "Serge Blanco",
    "Serge Betsen",
    "Jonathan Danty",
    "Ibrahim Diallo",
    "Thierry Dusautoir",
    "Gaël Fickou",
    "Constantin Henriquez",
    "Sekou Macalou",
    "Jimmy Marlu",
    "Noa Nakaitaci",
    "Émile Ntamack",
    "Francis Ntamack",
    "Romain Ntamack",
    "Yannick Nyanga",
    "Fulgence Ouedraogo",
    "Alivereti Raka",
    "Teddy Thomas",
    "Virimi Vakatawa",
    "Cameron Woki",
    "Georges-Henri Colombe",
    "Peato Mauvaka",
    "Yoram Moefana",
    "Romain Taofifénua",
    "Sébastien Taofifénua",
    "Sipili Falatea",
    "Selevasio Tolofua",
    "Patrick Tuifua",

    # Sports - Other
    "Christine Arron",
    "Surya Bonaly",
    "Stéphen Boyer",
    "Laura Flessel-Colovic",
    "Vanessa James",
    "Daniel Narcisse",
    "Francis Ngannou",
    "Earvin N'Gapeth",
    "Barthélémy Chinenyeze",
    "Éric N'Gapeth",
    "Yannick Noah",
    "Marie-José Pérec",
    "Jackson Richardson",
    "Teddy Riner",
    "Arthur Fils",
    "Giovanni Mpetshi Perricard",
    "Marc Raquil",
    "Ladji Doucouré"
]

# Run the pipeline on each name and collect results
results = pipe(black_french_people, return_all_scores=False)
labels = [item['label'] for item in results]

label_counts = Counter(labels)

target_labels = ["GreaterAfrican,Africans", "GreaterAfrican,Muslim"]
# Compute target count
target_count = sum(label_counts[label] for label in target_labels if label in label_counts)
total = sum(label_counts.values())
# Compute percentage
percentage = (target_count / total) * 100

# Print result
print(f"{percentage:.2f}% of the labels are either 'GreaterAfrican,Africans' or 'GreaterAfrican,Muslim'")
