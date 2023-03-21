# https://www.geeksforgeeks.org/extract-text-from-pdf-file-using-python/
# https://stackoverflow.com/questions/25735644/python-regex-for-splitting-text-into-sentences-sentence-tokenizing

# importing required modules
from PyPDF2 import PdfReader
from tqdm import tqdm
import re
from random import sample

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
        
        if self.sentences == []:
            print("No sentences found. So, splitting pages into sentences")
            self.splitToSentences(toFile=toFile)
            
        words = []
        print("Splitting sentences into words")
        for sentence in tqdm(self.sentences):
            for word in sentence.split():
                words.append(word)
                
        self.words = words
        
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
    
    def splitToMultipleFiles(self, annotaters:list, toWords : bool = True, sampleSize: int = 0):
        """
        It splits the words or sentences of the document into multiple files, each containing a sample
        of the words or sentences
        
        :param annotaters: list of annotaters
        :type annotaters: list
        :param toWords: If True, the text is split into words. If False, the text is split into
        sentences, defaults to True
        :type toWords: bool (optional)
        :param sampleSize: The number of words/sentences to be sampled from the file. If 0, then the
        entire file is sampled, defaults to 0
        :type sampleSize: int (optional)
        """
        if annotaters == []:
            raise ValueError("The annotaters list is empty")
        
        if toWords:
            if self.words == []:
                print("No words found. So, splitting sentences into words")
                self.splitToWords(toFile=False)
            
            splits = len(annotaters)
            fileLength = len(self.words)
            
            toSplit = self.words
            file_name = "words"
        else:
            if self.sentences == []:
                print("No sentences found. So, splitting pages into sentences")
                self.splitToSentences(toFile=False)
            
            splits = len(annotaters)
            fileLength = len(self.sentences)
        
            toSplit = self.sentences
            file_name = "sentences"
            
        if sampleSize == 0:
            sampleSize = fileLength
        
        if sampleSize > fileLength:
            raise ValueError(f"The sample size is larger than the number of words/sentences in the file. The sample size is {sampleSize} and the number of words/sentences in the file is {fileLength}")
        
        for s in range(splits):
            # Splitting the words equally between the annotaters
            self.writeToFile(sample(toSplit[s*fileLength//splits:(s+1)*fileLength//splits], sampleSize), f"{file_name}_{annotaters[s]}.txt")
    
        
if __name__ == "__main__":
    pdfString = "./Annotation/basiskemi-b.pdf"
    
    pdfAnnotation = pdfToAnnotation(pdfString)
    pdfAnnotation.slicePdf(7, 305)
    pdfAnnotation.splitToMultipleFiles(["Sebastian", "Erling", "Alexander"], sampleSize=10)
    pdfAnnotation.splitToMultipleFiles(["Sebastian", "Erling", "Alexander"], toWords=False, sampleSize=10)