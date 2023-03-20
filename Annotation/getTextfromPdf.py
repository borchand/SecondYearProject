# https://www.geeksforgeeks.org/extract-text-from-pdf-file-using-python/
# https://stackoverflow.com/questions/25735644/python-regex-for-splitting-text-into-sentences-sentence-tokenizing

# importing required modules
from PyPDF2 import PdfReader
from tqdm import tqdm
import re

# It takes a pdf file as a string and creates a PdfReader object
class pdfToAnnotation():
    def __init__(self, pdfString: str):
        """
        The function takes a pdf file as a string and creates a PdfReader object. It then sets the
        sliceFrom and sliceTo variables to 0 and the number of pages in the pdf file, respectively. It
        also creates two empty lists, sentences and words
        
        :param pdfString: The string of the pdf file
        :type pdfString: str
        """
        self.pdfReader = PdfReader(pdfString)
        self.sliceFrom = 0
        self.sliceTo =  len(self.pdfReader.pages)
        self.sentences = []
        self.words = []
    
    def slicePdf(self, f:int = 0, t:int = 0):
        """
        > It takes two arguments, `f` and `t`, and sets the `sliceFrom` and `sliceTo` attributes of the
        `Pdf` object to the values of `f` and `t` respectively
        
        :param f: int = 0, t: int = 0, defaults to 0
        :type f: int (optional)
        :param t: The number of pages to slice from the end of the PDF, defaults to 0
        :type t: int (optional)
        """
        if t <= f:
            t = len(self.pdfReader.pages)
        
        print(f"Original page count {self.sliceTo - self.sliceFrom}")
        
        self.sliceTo = t
        self.sliceFrom = f
        
        print(f"New page count {self.sliceTo - self.sliceFrom}")
        
    def splitToSentences(self, toFile: bool = True):
        """
        It splits the text into sentences and removes all newline characters from the sentence
        
        :param toFile: If True, the sentences are written to a file, defaults to True
        :type toFile: bool (optional)
        :return: A list of sentences.
        """
        sentences = []
        print("Splitting pages into sentences")
        for page in tqdm(self.pdfReader.pages[self.sliceFrom:self.sliceTo]):
            # Splitting the text into sentences.
            for sentence in re.split("(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?)\s", page.extract_text()):
                
                # It removes all newline characters from the sentence.
                sentence = re.sub("\n", "", sentence)
                # Checking if the sentence is a number. If it is not, it is appended to the list of
                # sentences.                
                if re.match(r"/^\d+$/", sentence.strip()) == None:
                    sentences.append(sentence)

        self.sentences = sentences
        
        if toFile:
            self.writeToFile(sentences, "sentences.txt")
        return sentences
    
    def splitToWords(self, toFile: bool = True):
        """
        It takes a list of sentences, splits each sentence into words, and returns a list of words
        
        :param toFile: bool = True, defaults to True
        :type toFile: bool (optional)
        :return: A list of words
        """
        words = []
        print("Splitting sentences into words")
        for sentence in tqdm(self.sentences):
            for word in sentence.split():
                words.append(word)
            
        if toFile:
            self.writeToFile(words, "words.txt")
        
        return words
    
    def writeToFile(self, lists: list, fileName: str):
        """
        It takes a list of strings and writes them to a file
        
        :param lists: list
        :type lists: list
        :param fileName: The name of the file you want to write to
        :type fileName: str
        """
        if not fileName.endswith(".txt"):
            raise ValueError("The file name must end with .txt")

        with open(fileName,'w') as sentencesFile:
            sentencesFile.write('\n'.join(lists))
         
    def getSentencesList(self):
        """
        It returns the sentences list.
        :return: The sentences list is being returned.
        """
        return self.sentences
        
if __name__ == "__main__":
    pdfString = "./Annotation/basiskemi-b.pdf"
    
    pdfAnnotation = pdfToAnnotation(pdfString)
    pdfAnnotation.slicePdf(7, 305)
    pdfAnnotation.splitToSentences()
    pdfAnnotation.splitToWords()