Metoda de generare consta in capturarea unei poze cu fata mea, cu ajutorul librariei opencv, folosirea detectyorului de fete "haarcascades" care este un algoritm care detecteaza marginile dintr-o poza, fiind special pentru formele unei fete. Porgramul dupa preia coordonatele dreptunghiului in care se afla fata si in functie de emotia pe care o dorim capturata apasam tasta specifica pe tastatura (ex: pentru angry apasam 'a') si poza decupata este redimensionata la 100x100 pixeli si salvata in folderul genrated/[numele emotiei] 
Parametrii folositi:
- IMAGE_SIZE: 100
- GRAYSCALE: pentru detectarea fetei
- VideoCapture: pentru salvarea cadrelor inregistrate de camera
Datele generate sunt relevante pentru antrenarea unui model `custom` care se poate adapta la emotiile faciale ale unui utilizator