import os, sys
import sqlite3
import faiss
import numpy as np
from tqdm import tqdm

from langchain_core.embeddings import Embeddings
from langchain_text_splitters.base import TextSplitter
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader

# typing
from typing import List, Tuple, Optional
from langchain_core.documents import Document

""" Constants """
DEFAULT_EMBEDDING_MODEL_NAME = 'text-embedding-3-large'
DEFAULT_EMBEDDIMG_DIMENSION = 3072

""" Hybrid Database (sqlite3 + faiss) """
class MyVectorDatabase:
    def __init__(self, 
                 database_path:str, 
                 database_name:str,
                 dimension:int = DEFAULT_EMBEDDIMG_DIMENSION,
                 verbose:bool = False,
                 **kwargs,
        ):  
        self.database_path = database_path
        self.database_name = database_name

        self._dimension = dimension
        self.verbose = verbose

        self._sqlite3_extension     = kwargs.get('sqlite3_extension', '_sqlite3.db')
        self._faiss_index_extension = kwargs.get('faiss_index_extension', '_faiss.index')

        self.conn = None
        self.cursor = None
        self.faiss_index = None
        self._sqlite3_table_name = kwargs.get('sqlite3_table_name', 'faiss_map')

    """ initialize """
    def _connect_sqlite3(self) -> 'MyVectorDatabase':
        self.conn = sqlite3.connect( self._full_sqlite3_path )
        self.cursor = self.conn.cursor()
        self.cursor.execute(f'''
            CREATE TABLE IF NOT EXISTS {self._sqlite3_table_name} (
                id INTEGER PRIMARY KEY,
                filepath TEXT,
                filename TEXT,
                chunk_number INTEGER,
                vector BLOB
            )
        ''')

        self.conn.commit()
        return self
    
    def _connect_faiss_index(self) -> 'MyVectorDatabase':
        try:
            self.faiss_index = faiss.read_index(self._full_faiss_index_path)
            self._dimension = self.faiss_index.d
        except:
            faiss_index = faiss.IndexFlatL2(self.dimension)
            self.faiss_index = faiss.IndexIDMap(faiss_index)
            faiss.write_index(self.faiss_index, self._full_faiss_index_path)
        return self
    
    def connect(self) -> 'MyVectorDatabase':
        if not os.path.exists(self.database_path):
            os.makedirs(self.database_path)
        
        self._connect_sqlite3()
        self._connect_faiss_index()

        if self.verbose:
            print(f"Database connected: {self.full_database_path}")
        return self
    
    def is_connected(self) -> bool:
        return self.conn is not None and self.cursor is not None and self.faiss_index is not None
    
    """ check """
    def _is_sqlite3_exist(self) -> bool:
        return os.path.exists( self._full_sqlite3_path )
    
    def _is_faiss_exist(self) -> bool:
        return os.path.exists( self._full_faiss_index_path )
    
    def is_database_exist(self) -> bool:
        return self._is_sqlite3_exist() and self._is_faiss_exist()
    
    def _get_faiss_ids(self) -> np.ndarray:
        stored_ids =  faiss.vector_to_array(self.faiss_index.id_map)
        return stored_ids
    
    def _get_sqlite3_ids(self) -> np.ndarray:
        self.cursor.execute('SELECT id FROM faiss_map')
        ids = self.cursor.fetchall()
        id_list = [id_tuple[0] for id_tuple in ids]
        return np.array(id_list)
    
    """ close """
    def close(self) -> 'MyVectorDatabase':
        if self.conn:
            self.conn.close()
        return self
    
    """ insert """
    def _insert_embeddings_to_faiss(self, 
                                    id:int,
                                    embedding:np.ndarray,
        ) -> 'MyVectorDatabase':
        """ insert embeddings to the FAISS index."""
        try:
            self.faiss_index.remove_ids( np.array([id], dtype=np.int64) )
        except:
            pass

        # insert embeddings
        embeddings = np.array([embedding], dtype='float32')
        ids = np.array([id], dtype='int64')

        # check the shape
        if embeddings.shape[1] != self.dimension:
            raise ValueError(f"Dimension mismatch: embedding dimension {embeddings.shape[1]} != faiss dimension {self.dimension}")

        self.faiss_index.add_with_ids( embeddings, ids )
        faiss.write_index(self.faiss_index, self._full_faiss_index_path)
        return self
    
    def _insert_metadata_to_sqlite(
                self,
                id:int,
                document_path:str,
                document_name:str,
                embedding:np.ndarray,
                chunk_number:int,
        ) -> 'MyVectorDatabase':
        """Add metadata to the SQLite database."""            
        # insert metadata
        self.cursor.execute(f"INSERT OR REPLACE INTO {self._sqlite3_table_name} (id, filepath, filename, chunk_number, vector) VALUES (?, ?, ?, ?, ?)",
                            (id, document_path, document_name, chunk_number, embedding.tobytes()))
        self.conn.commit()
        return self
    
    def _get_max_id(self) -> int:
        """Get the maximum id from the SQLite database."""
        self.cursor.execute(f"SELECT MAX(id) FROM {self._sqlite3_table_name}")
        row = self.cursor.fetchone()
        return row[0] if row[0] else 0
    
    def insert( self,
                document_path:str,
                document_name:str,
                embedding:np.ndarray,
                chunk_number:int,
        ) -> 'MyVectorDatabase':
        """Insert the document to the database."""
        id = self._get_max_id() + 1
        if self.is_file_in_database(document_path = document_path, document_name = document_name, chunk_number = chunk_number):
            id = self.cursor.execute(f"SELECT id FROM {self._sqlite3_table_name} WHERE filepath=? AND filename=? AND chunk_number=?", (document_path, document_name, chunk_number)).fetchone()[0]
        
        self._insert_embeddings_to_faiss(id=id, embedding=embedding)
        self._insert_metadata_to_sqlite(id=id, document_path=document_path, document_name=document_name, embedding=embedding, chunk_number=chunk_number)
        return self

    """ query """
    def is_file_in_database(
            self,
            document_path:str,
            document_name:str,
            chunk_number:Optional[int] = None,
        ) -> bool:
        """Check if the document is in the database."""
        if chunk_number is not None:
            self.cursor.execute(f"SELECT * FROM {self._sqlite3_table_name} WHERE filepath=? AND filename=? AND chunk_number=?", (document_path, document_name, chunk_number))
        else:
            self.cursor.execute(f"SELECT * FROM {self._sqlite3_table_name} WHERE filepath=? AND filename=?", (document_path, document_name))
        row = self.cursor.fetchone()
        return row is not None
                            
    def _query_faiss_index(self, 
                     embedding:np.ndarray,
                     k:int = 5,
        ) -> Tuple[np.ndarray, np.ndarray]:
        """ query embeddings from the FAISS index."""
        D, I = self.faiss_index.search(embedding, k)
        return D, I
    
    def _query_metadata(self,
                        id:int,
        ) -> Tuple[str, str, int, int, np.ndarray]:
        """ query metadata from the SQLite database."""
        self.cursor.execute(f'SELECT * FROM {self._sqlite3_table_name} WHERE id = ?', (id,))
        row = self.cursor.fetchone()

        if row is None:
            raise ValueError(f"ID {id} (type={type(id)}) is not in the database (table={self._sqlite3_table_name}).")
        
        id, document_path, document_name, chunk_number, embedding = row

        embedding = np.frombuffer(embedding, dtype=np.float32)
        return id, document_path, document_name, chunk_number, embedding
    
    def query(self,
              embedding:np.ndarray,
              k:int = 5,
              get_distance_and_index:bool = False,
        ) -> List[Tuple[str, str, int, int, np.ndarray]]:
        D, I = self._query_faiss_index(embedding, k)
        
        results = []
        for i in range(k):
            id = int( I[0][i] )
            id, document_path, document_name, chunk_number, embedding = self._query_metadata(id)
            results.append((document_path, document_name, chunk_number, embedding))
        
        if get_distance_and_index:
            return results, D[0], I[0]
        return results
                        
    """ property """
    def _get_datanumber_in_sqlite3(self) -> int:
        """Count the number of records in the vectors table."""
        self.cursor.execute(f"SELECT COUNT(*) FROM {self._sqlite3_table_name}")
        count = self.cursor.fetchone()[0]
        return count

    def _get_datanumber_in_faiss_index(self) -> int:
        """Count the number of records in the faiss index."""
        return self.faiss_index.ntotal
    
    def check_datanumber_in_database(self) -> bool:
        """Check the number of records in the database."""
        return self._get_datanumber_in_sqlite3() == self._get_datanumber_in_faiss_index()

    def __len__(self) -> int:
        """Count the number of records in the vectors table."""
        return self._get_datanumber_in_sqlite3()

    @property
    def full_database_path(self):
        return os.path.join(self.database_path, self.database_name)

    @property
    def _full_sqlite3_path(self):
        return self.full_database_path + self._sqlite3_extension
    
    @property
    def _full_faiss_index_path(self):
        return self.full_database_path + self._faiss_index_extension
    
    @property
    def dimension(self):
        return self._dimension


DEFAULT_CHUNK_SIZE = 2000
DEFAULT_CHUNK_OVERLAP = 500
DEFAULT_TEXT_SPITTER = RecursiveCharacterTextSplitter(chunk_size = DEFAULT_CHUNK_SIZE, chunk_overlap = DEFAULT_CHUNK_OVERLAP)
class MyVectorDatabaseWithEmbeddedModel(MyVectorDatabase):
    def __init__(self, 
                 embeddings_model:Embeddings,
                 database_path:str, 
                 database_name:str,
                 dimension:int = DEFAULT_EMBEDDIMG_DIMENSION,
                 text_splitter:Optional[TextSplitter] = DEFAULT_TEXT_SPITTER,
                 verbose:bool = False,
                 **kwargs,
        ):  
        super().__init__(database_path, database_name, dimension, verbose, **kwargs)
        self.embeddings_model = embeddings_model
        self.text_splitter = text_splitter

    def process_pdf_to_chunks(
            self, 
            pdf_filepath:str,
            pdf_filename:str,
        ) -> List[Document]:
        """Process a document and generate chunks."""
        # load and split the document
        loader = PyPDFLoader( os.path.join(pdf_filepath, pdf_filename) )
        chunks = loader.load_and_split(self.text_splitter)
        return chunks
    
    def process_pdf_to_embedding(
            self, 
            pdf_filepath:str,
            pdf_filename:str,
            progress:bool = False,
            skip_if_exist:bool = False,
        ) -> np.ndarray:
        """Process a document and generate embeddings."""
        # load and split the document
        chunks = self.process_pdf_to_chunks(pdf_filepath, pdf_filename)
        if skip_if_exist:
            chunks = [chunk for chunk_number, chunk in enumerate(chunks) if not self.is_file_in_database(document_path = pdf_filepath, document_name = pdf_filename, chunk_number = chunk_number)]

        # generate embeddings
        iterator = tqdm(chunks) if progress else chunks
        embeddings = [] 
        for chunk in iterator :
            # append the embedding
            embeddings.append( self.embeddings_model.embed_query(chunk.page_content) )
        return np.array( embeddings )

    def is_pdf_file_in_database(
            self,
            pdf_filepath:str,
            pdf_filename:str,
        ) -> bool:
        """Check if the document is in the database."""
        chunks = self.process_pdf_to_chunks(pdf_filepath, pdf_filename)
        return all([self.is_file_in_database(document_path = pdf_filepath, document_name = pdf_filename, chunk_number = chunk_number) for chunk_number, _ in enumerate(chunks)])
    
    def insert_pdf_file(             
            self, 
            pdf_filepath:str,
            pdf_filename:str,
            progress:bool = False,
            force:bool = False,
        ) -> 'MyVectorDatabaseWithEmbeddedModel':
        """Insert the document to the database."""
        # process the document
        embeddings = self.process_pdf_to_embedding(pdf_filepath  = pdf_filepath, 
                                                   pdf_filename  = pdf_filename, 
                                                   progress      = progress,
                                                   skip_if_exist = not force)

        # insert the document
        for chunk_number, embedding in enumerate(embeddings):
            self.insert(document_path = pdf_filepath, 
                        document_name = pdf_filename, 
                        embedding     = embedding, 
                        chunk_number  = chunk_number)
        return self

    def query(
             self, 
             query:str,
             k:int = 5,
             get_distance_and_index:bool = False,
             format:Optional[str] = None,
        ) -> List[Document]:
        """Query the database."""
        format = 'chunks' if format is None else format

        # generate query embeddings
        embedding = self.embeddings_model.embed_query(query)
        embedding = np.array([embedding], dtype='float32')

        # query the database
        results, D, I = super().query(embedding, k, get_distance_and_index = True)

        answers = []
        for result in results:
            document_path, document_name, chunk_number, _ = result

            if format.upper() in ('CHUNKS', 'CHUNK'):
                chunks = self.process_pdf_to_chunks(document_path, document_name)
                answers.append(chunks[chunk_number])
            elif format.upper() in ('EMBEDDINGS', 'EMBEDDING'): 
                answers.append(embedding)
            elif format.upper() in ('DOCUMENTS', 'DOCUMENT', 'DOC'):
                docs = PyPDFLoader( os.path.join(document_path, document_name) ).load()
                answers.append( docs )
            elif format.upper() in ('PAGE', 'PAGES'):
                chunk = self.process_pdf_to_chunks(document_path, document_name)[chunk_number]
                docs = PyPDFLoader( os.path.join(document_path, document_name) ).load()
                answers.append( docs[ chunk.metadata['page'] ] )
            else:
                raise ValueError(f"Unknown format: {format}")

        if get_distance_and_index:
            return answers, D, I
        return answers
    


