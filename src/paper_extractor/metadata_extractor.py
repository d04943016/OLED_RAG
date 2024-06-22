# packages
import os, sys
from tqdm import trange
import re
import json
import sqlite3
from collections import defaultdict
import pandas as pd

# langchain 
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser, PydanticOutputParser
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.exceptions import OutputParserException
from langchain_community.document_loaders import PyPDFLoader

# typing
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.output_parsers.base import BaseOutputParser
from typing import Optional, List, Dict, Any, Union

""" extractor prompts """
META_DATA_NAMES = ['title', 'authors', 'publish year', 'publish month', 'publish day',
                   'abstract', 'affliation', 'emails', 
                   'paper publisher', 'DOI', 'Journal', 'First author(s)', 'keywords',
                   'citation format', 'section titles', 'figure captions', 'table captions']

# prompt_template is a prompt template to extract metadata from a document
prompt_template = PromptTemplate(
    template='''You are a paper metadata extractor from the following document. '''
             '''The document information is between [Document Start] and [Document End]. '''
             '''Most of metadata information should be in the first page of the document. (But not always true) '''
             '''You will only read one page of the document. '''
             '''Now I will give you page {page_number} of the document. '''
             '''[Document Start]\n '''
             '''{document}\n\n '''
             '''[Document End]\n\n '''
             '''{format_instructions}\n '''
             '''Please ignore or remove any special characters, including but not limited to:\n '''
             '''- Hexadecimal escape sequences (e.g., \\x00, \\x01, etc.)\n '''
             '''- Control characters (e.g., \\n, \\r, \\t, etc.)\n '''
             '''- Other non-visible characters\n ''',
    input_variables=['meta_data_names', 'document', 'format_instructions'],
)

# default_output_parser is the default output parser for the model
DEFAULT_JSON_OUTPUT_PARSER = JsonOutputParser()

# AcademicPaperMetadata is the Pydantic model for the metadata extracted from the document
not_found_template = PromptTemplate( 
        template = 'If you cannot find the information, please return {default_value}.', 
        input_variables=['default_value'],
)
not_found_NA = not_found_template.partial(default_value='N/A')
not_found_None = not_found_template.partial(default_value='None')
not_found_list = not_found_template.partial(default_value='[] (empty list)')
not_found_dict = not_found_template.partial(default_value='{} (empty dictionary)')
not_found_list_with_length = not_found_template.partial(default_value='a list with "N/A" elements for not found information')

class AcademicPaperMetadata(BaseModel):
    """ The Pydantic model for the metadata extracted from the document. """
    title: str                 = Field(description = f'The title of the paper. {not_found_NA.format()}')
    authors:List[str]          = Field(description = f'The authors of the paper. The order of the authors should be preserved. ')
    publish_year: str          = Field(description = f'The year the paper was published. {not_found_NA.format()}')
    publish_month: str         = Field(description = f'The month the paper was published. Use number instead of word. {not_found_NA.format()}')
    publish_day: str           = Field(description = f'The day the paper was published. {not_found_NA.format()}')
    abstract: str              = Field(description = f'The abstract of the paper. If must be in the first page of the paper {not_found_NA.format()}')
    affliations: List[str]     = Field(description = f'The affliation of each authors. The length should be the same as the authors. {not_found_list_with_length.format()}')
    emails: Dict[str, str]     = Field(description = f'The emails of the communication authors. The keys should be the authors and the values should be the emails. {not_found_dict.format()}')
    paper_publisher: str       = Field(description = f'The publisher of the paper. {not_found_NA.format()}')
    DOI: str                   = Field(description = f'The DOI of the paper. If is is an URL, please use full URL. {not_found_NA.format()}')
    Journal: str               = Field(description = f'The journal that the paper was published in. {not_found_NA.format()}')
    first_authors: List[str]   = Field(description = f'''The first authors of the paper. '''
                                                     f'''Basically, the first author is the author in the first place of the authors list. '''
                                                     f'''If there are multiple authors who contributed equally, the paper would explicitly mention that.'''
                                                     f'''The order should be preserved. {not_found_list.format()}''')
    keywords: List[str]        = Field(description = f'The keywords of the paper. If there is no keywords, please extract the keywords from the abstract. If no keywords are found, please return an empty list.')
    citation_format: str       = Field(description = f'The citation format of the paper [APA format]. {not_found_NA.format()}')
    section_titles: List[str]  = Field(description = f'The section titles of the paper. {not_found_list.format()}')
    figure_captions: Dict[str, List[str]] = Field(description = f'''The figure captions of the paper.'''
                                                                f'''The caption text associated with the figure. '''
                                                                f'''The caption text typically follows the word "Figure" or 'Fig.' and a number (e.g., Figure 1/Fig. 1) and provides a brief description or explanation of the figure.'''
                                                                f'''Do not separate the sub-images into different captions.'''
                                                                f'''The key should be the figure number or id and the value is a list of the caption text(s).'''
                                                                f'''Please keep figure number. {not_found_dict.format()}''')
    table_captions: Dict[str, List[str]]  = Field(description = f'The table captions of the paper. '''
                                                                f'''The caption text associated with the table. '''
                                                                f'''The caption text typically follows the word "Table" and a number (e.g., Table 1) and provides a brief description or explanation of the figure.'''
                                                                f'''Please keep table number. {not_found_dict.format()}')''')

def get_empty_academicpapermetadata() -> AcademicPaperMetadata:
    """ Get the empty AcademicPaperMetadata object. """
    return AcademicPaperMetadata(
        title = 'N/A',
        authors = [],
        publish_year = 'N/A',
        publish_month = 'N/A',
        publish_day = 'N/A',
        abstract = 'N/A',
        affliations = [],
        emails = {},
        paper_publisher = 'N/A',
        DOI = 'N/A',
        Journal = 'N/A',
        first_authors = [],
        keywords = [],
        citation_format = 'N/A',
        section_titles = [],
        figure_captions = defaultdict(list),
        table_captions = defaultdict(list),
    )

DEFAULT_PARSER = PydanticOutputParser(pydantic_object = AcademicPaperMetadata)

# prompt_template_with_output_parser is the prompt template with the default output parser
prompt_template_with_output_format_instructions = prompt_template.partial(format_instructions=DEFAULT_PARSER.get_format_instructions())

""" metadata extracting functions """
def clean_invalid_json(invalid_json):
    # Remove unwanted characters and clean up the JSON string
    print( str(invalid_json) )
    cleaned_json = re.sub(r'\\x[0-9A-Fa-f]{2}', '', invalid_json)  # Remove hexadecimal escape sequences
    cleaned_json = re.sub(r'[\x00-\x1f\x7f-\x9f]', '', cleaned_json)  # Remove control characters
    cleaned_json = re.sub(r'\s+', ' ', cleaned_json)  # Replace multiple spaces/newlines with a single space
    cleaned_json = cleaned_json.replace('\n', ' ')  # Replace newlines with spaces
    cleaned_json = re.sub(r'(\s,)+', ',', cleaned_json)  # Remove spaces before commas
    cleaned_json = re.sub(r'(\s)+:', ':', cleaned_json)  # Remove spaces before colons
    cleaned_json = re.sub(r':(\s)+', ':', cleaned_json)  # Remove spaces after colons

    try:
        json_data = json.loads(cleaned_json)
    except json.JSONDecodeError as e:
        msg = f"{cleaned_json}"
        raise OutputParserException(msg, llm_output=cleaned_json) from e
    return json_data

def get_metadata_one_page(
        document: str, 
        chat_model:BaseChatModel,
        page_number:int, 
        prompt_template: PromptTemplate = prompt_template_with_output_format_instructions,
        output_parsor: Optional[BaseOutputParser] = DEFAULT_PARSER,
        try_count: int = 1,
) -> Any:
    """ 
    Extract metadata from a document using the chat model. 
    
    usage
    -----
    metadata = get_metadata_one_page(document, chat_model, page_number, prompt_template, output_parsor)

    Parameters
    ----------
    document : str
        The document to extract metadata from.
    
    chat_model : BaseChatModel
        The chat model to use to extract metadata.
    
    page_number : int
        The page number of the document.
    
    prompt_template : PromptTemplate
        The prompt template to use to extract metadata.
    
    output_parsor : Optional[BaseOutputParser]
        The output parser to use to parse the output of the model.
    
    try_count : int
        The number of times to try to extract metadata from the document.
    
    Returns
    -------
    Any
        The metadata extracted from the document.

    """
    prompt_values = prompt_template.format(page_number=page_number, document=document)
    try_count = 1 if try_count < 1 else try_count
    
    for _ in range(try_count):
        try:
            message = chat_model.invoke(prompt_values)
            if output_parsor:
                try:
                    return output_parsor.invoke(message)
                except Exception as e:
                    json_data = clean_invalid_json(e.args[0][28:-3])
                    print(json_data)
                    return AcademicPaperMetadata(**json_data)
            return message
        except Exception as e:
            pass
    return get_empty_academicpapermetadata()

def merge_academic_paper_metadata(metadata: List[AcademicPaperMetadata]) -> AcademicPaperMetadata:
    """ 
    Merge the metadata of the documents into one metadata. 
    
    usage
    -----
    metadata = merge_academic_paper_metadata(metadata)

    Parameters
    ----------
    metadata : List[AcademicPaperMetadata]
        The list of metadata to merge.

    Returns
    -------
    AcademicPaperMetadata
        The merged metadata.

    """
    metadata_dict = get_empty_academicpapermetadata().dict()

    for data in metadata:
        # title, publish_year, publish_month, publish_day, abstract, paper_publisher, DOI, Journal, citation_format
        NA_tags = ['title', 'publish_year', 'publish_month', 'publish_day', 'abstract', 'paper_publisher', 'DOI', 'Journal', 'citation_format']
        for tag in NA_tags:
            if metadata_dict[tag] == 'N/A' and getattr(data, tag) != 'N/A':
                metadata_dict[tag] = getattr(data, tag)
        
        # authors, affliations, emails, first_authors, keywords
        empty_list_tags = ['authors', 'affliations', 'first_authors', 'keywords']
        for tag in empty_list_tags:
            if getattr(data, tag) and len(metadata_dict[tag]) == 0:
                metadata_dict[tag] = getattr(data, tag)

        # emails
        if getattr(data, 'emails') and len(metadata_dict['emails']) == 0:
            for key, value in getattr(data, 'emails').items():
                metadata_dict['emails'][key] = value

        # section_titles
        empty_list_tags = ['section_titles',]
        for tag in empty_list_tags:
            if getattr(data, tag):
                metadata_dict[tag] += getattr(data, tag)
            
        # figure_captions, table_captions
        empty_dict_tags = ['figure_captions', 'table_captions']
        for tag in empty_dict_tags:
            if getattr(data, tag):
                for key, value in getattr(data, tag).items():
                    if key not in metadata_dict[tag]:
                        metadata_dict[tag][key] = [] 
                    metadata_dict[tag][key] += value
        
    return AcademicPaperMetadata(**metadata_dict)

def get_metadata(
        doucments: List[str],
        chat_model:BaseChatModel,
        prompt_template: PromptTemplate = prompt_template_with_output_format_instructions,
        output_parsor: Optional[BaseOutputParser] = DEFAULT_PARSER,
        merge: bool = True,
        try_count_per_document: int = 1,
        verbose: bool = True,
) -> Any:
    """ 
    Extract metadata from a list of documents using the chat model. 
    
    usage
    -----
    metadata = get_metadata(doucments, chat_model, prompt_template, output_parsor, merge, auto_stop, verbose)

    Parameters
    ----------
    doucments : List[str]
        The list of documents to extract metadata from.
    
    chat_model : BaseChatModel
        The chat model to use to extract metadata.
    
    prompt_template : PromptTemplate
        The prompt template to use to extract metadata.
    
    output_parsor : Optional[BaseOutputParser]
        The output parser to use to parse the output of the model.

    merge : bool
        Whether to merge the metadata of the documents into one metadata.
    
    try_count_per_document : int
        The number of times to try to extract metadata from each document.
    
    verbose : bool  
        Whether to show the progress bar.

    Returns
    -------
    Any
        The metadata extracted from the documents.

    """
    metadata = []
    iterator = trange(len(doucments)) if verbose else range(len(doucments))

    for ii in iterator:
        page_number = ii + 1
        data = get_metadata_one_page( document        = doucments[ii], 
                                      chat_model      = chat_model, 
                                      page_number     = page_number, 
                                      prompt_template = prompt_template, 
                                      output_parsor   = output_parsor,
                                      try_count       = try_count_per_document)
        metadata.append(data)
    
    if merge:
        return merge_academic_paper_metadata(metadata)
    return metadata

""" database """
def create_new_database(database_path:str, database_name:str) -> sqlite3.Connection:
    if not os.path.exists(database_path):
        os.makedirs(database_path)
    
    conn = sqlite3.connect( os.path.join(database_path, f'{database_name}.db') )
    cursor = conn.cursor()

    cursor.execute('''
        CREATE TABLE IF NOT EXISTS metadata (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            title TEXT,
            authors TEXT,
            publish_year TEXT,
            publish_month TEXT,
            publish_day TEXT,
            abstract TEXT,
            affliations TEXT,
            emails TEXT,
            paper_publisher TEXT,
            DOI TEXT,
            Journal TEXT,
            first_authors TEXT,
            keywords TEXT,
            citation_format TEXT,
            section_titles TEXT,
            figure_captions TEXT,
            table_captions TEXT,
            filepath TEXT,
            filename TEXT
        );
    ''')
    conn.commit()
    return conn
    
def get_database_connection(
        database_path:str,
        database_name:str,
        create_new_if_not_exist:bool = False,
    ) -> sqlite3.Connection:
    """
    get the database connection

    usage
    -----
    conn = get_database_connection(database_path, database_name, create_new_if_not_exist)

    Parameters
    ----------
    database_path : str
        The path of the database.

    database_name : str
        The name of the database.

    create_new_if_not_exist : bool
        Whether to create a new database if it does not exist.

    Returns
    -------
    sqlite3.Connection
        The connection to the database.
        
    """
    if not os.path.exists( os.path.join(database_path, f'{database_name}.db') ):
        if create_new_if_not_exist:
            return create_new_database(database_path = database_path, database_name = database_name)
        raise FileNotFoundError(f'The database {database_name}.db does not exist in the path {database_path}')
    conn = sqlite3.connect( os.path.join(database_path, f'{database_name}.db') )
    return conn

def insert_metadata(conn:sqlite3.Connection,
                    metadata:AcademicPaperMetadata, 
                    filepath:str, 
                    filename:str,
                    verbose:bool = True, 
    ) -> bool:
    # 检查文件是否已经存在
    cursor = conn.cursor()
    cursor.execute('SELECT * FROM metadata WHERE filename=? AND filepath=?', (filename, filepath))
    if cursor.fetchone():
        if verbose:
            print("Metadata already extracted for this file.")
        return False

    # 插入元数据到数据库
    cursor.execute('''
        INSERT INTO metadata (
            title, authors, publish_year, publish_month, publish_day, abstract, affliations, emails, 
            paper_publisher, DOI, Journal, first_authors, keywords, citation_format, section_titles, 
            figure_captions, table_captions, filepath, filename
        ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
    ''', (metadata.title, 
          str(metadata.authors), 
          metadata.publish_year, 
          metadata.publish_month, 
          metadata.publish_day, 
          metadata.abstract, 
          str(metadata.affliations), 
          str(metadata.emails), 
          metadata.paper_publisher,
          metadata.DOI, 
          metadata.Journal, 
          str(metadata.first_authors), 
          str(metadata.keywords), 
          metadata.citation_format, 
          str(metadata.section_titles), 
          str(metadata.figure_captions), 
          str(metadata.table_captions), 
          filepath, 
          filename))
    
    conn.commit()
    if verbose:
        print("Metadata inserted successfully.")
    return True

def get_metadata_from_database_with_filepath_and_name(
        conn:sqlite3.Connection, 
        pdf_filepath:str, 
        pdf_filename:str, 
        get_id:bool = False,
        verbose:bool = True,
    ) -> Union[ Union[AcademicPaperMetadata,int],None]:
    """
    Get the metadata from the database given with filepath and filename.

    usage
    -----
    metadata = get_metadata_from_database_with_filepath_and_name(conn, pdf_filepath, pdf_filename, get_id = False, verbose = True)
    metadata, id = get_metadata_from_database_with_filepath_and_name(conn, pdf_filepath, pdf_filename, get_id = True, verbose = True)

    Parameters
    ----------
    conn : sqlite3.Connection
        The connection to the database.

    pdf_filepath : str
        The filepath of the pdf file.

    pdf_filename : str
        The filename of the pdf file.

    get_id : bool
        Whether to get the id of the metadata.

    verbose : bool
        Whether to show the progress bar.

    Returns
    -------
    Union[ Union[AcademicPaperMetadata,int],None]
        The metadata extracted from the database. If get_id is True, return the metadata and the id of the metadata.

    """
    cursor = conn.cursor()
    cursor.execute('SELECT * FROM metadata WHERE filename=? AND filepath=?', (pdf_filename, pdf_filepath))
    data = cursor.fetchone()
    if data:
        id = data[0]
        metadata = AcademicPaperMetadata(
            title         = data[1],
            authors       = eval(data[2]),
            publish_year  = data[3],
            publish_month = data[4],
            publish_day   = data[5],
            abstract      = data[6],
            affliations   = eval(data[7]),
            emails        = eval(data[8]),
            paper_publisher = data[9],
            DOI           = data[10],
            Journal       = data[11],
            first_authors = eval(data[12]),
            keywords      = eval(data[13]),
            citation_format = data[14],
            section_titles  = eval(data[15]),
            figure_captions = eval(data[16]),
            table_captions  = eval(data[17]),
        )
        if verbose:
            print("Metadata retrieved successfully from the database.")
        
        if get_id:
            return metadata, id
        return metadata

    if verbose:
        print("Metadata not found in the database.")
    return None

def is_in_database(
        conn:sqlite3.Connection, 
        pdf_filepath:str, 
        pdf_filename:str, 
        verbose:bool = True,
    ) -> bool:
    """
    Check if the metadata is already in the database given with filepath and filename.
    
    usage
    -----
    is_in_database(conn, pdf_filepath, pdf_filename, verbose)
    
    Parameters
    ----------
    conn : sqlite3.Connection
        The connection to the database.

    pdf_filepath : str
        The filepath of the pdf file.

    pdf_filename : str
        The filename of the pdf file.

    verbose : bool
        Whether to show the progress bar.

    Returns
    -------
    bool
        Whether the metadata is already in the database.    
        
    """
    cursor = conn.cursor()
    cursor.execute('SELECT * FROM metadata WHERE filename=? AND filepath=?', (pdf_filename, pdf_filepath))
    data = cursor.fetchone()
    if data:
        if verbose:
            print("Metadata already extracted for this file.")
        return True
    if verbose:
        print("Metadata not found in the database.")
    return False

def convert_database_to_dataframe(
        conn:sqlite3.Connection,    
    ):
    # 执行SQL查询并将结果转换为Pandas DataFrame
    query = "SELECT * FROM metadata"
    df = pd.read_sql_query(query, conn)
    return df

def get_database(
        database_path:str,
        database_name:str,
        create_new_if_not_exist:bool = False,
        type:str = 'dataframe',
    ) -> Any:
    """
    Get the database connection or dataframe.
    
    usage
    -----
    conn = get_database(database_path, database_name, create_new_if_not_exist, type)

    Parameters
    ----------
    database_path : str
        The path of the database.
    
    database_name : str
        The name of the database.
    
    create_new_if_not_exist : bool
        Whether to create a new database if it does not exist.

    type : str
        The type of the return value. It can be 'dataframe' or 'connection'.

    Returns
    -------
    Any
        The database connection or dataframe.

    """
    conn = get_database_connection(database_path = database_path, 
                                   database_name = database_name, 
                                   create_new_if_not_exist = create_new_if_not_exist)

    if type.upper() in ('DATAFRAME', 'DF'):
        return convert_database_to_dataframe(conn)
    return conn


""" merge """
def get_metadata_from_database_or_pdf(
        pdf_filepath:str, 
        pdf_filename:str, 
        conn:sqlite3.Connection, 
        chat_model:BaseChatModel,
        verbose:bool = True,
    ):
    """
    get metadata from database or pdf file

    usage
    -----
    metadata = get_metadata_from_database_or_pdf(pdf_filepath, pdf_filename, conn, chat_model, verbose)

    Parameters
    ----------
    pdf_filepath : str
        The filepath of the pdf file.
    
    pdf_filename : str
        The filename of the pdf file.

    conn : sqlite3.Connection
        The connection to the database.

    chat_model : BaseChatModel
        The chat model to use to extract metadata.

    verbose : bool
        Whether to show the progress bar.

    Returns
    -------
    Any
        The metadata extracted from the pdf file.

    """
    # get metadata from database
    metadata = get_metadata_from_database_with_filepath_and_name(
        conn = conn,
        pdf_filepath = pdf_filepath,
        pdf_filename = pdf_filename,
        get_id = False,
        verbose = verbose, 
    )

    # extract metadata from pdf if not exists
    if metadata is None:
        loader = PyPDFLoader(file_path=os.path.join(pdf_filepath, pdf_filename))
        docs = loader.load()
        if verbose:
            print(f'The PDF file has {len(docs)} pages')

        metadata = get_metadata(docs, chat_model, verbose = verbose)
        insert_metadata(conn = conn, metadata = metadata, filepath = pdf_filepath,  filename = pdf_filename, verbose = verbose)
    return metadata




