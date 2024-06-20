from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser, PydanticOutputParser
from langchain_core.pydantic_v1 import BaseModel, Field
from tqdm import trange

# typing
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.output_parsers.base import BaseOutputParser
from typing import Optional, List, Dict, Any

META_DATA_NAMES = ['title', 'authors', 'publish year', 'abstract', 'affliation', 'emails', 
                   'paper publisher', 'DOI', 'Journal', 'First author(s)', 'keywords']

# prompt_template is a prompt template to extract metadata from a document
prompt_template = PromptTemplate(
    template='''You are a paper metadata extractor to extract {meta_data_names} from the following document.'''
             '''The document information is between [Document Start] and [Document End].'''
             '''The most metadata information should be in the first page of the document.'''
             '''You will only read one page of the document. If you need to read the next page, please return True.'''
             '''Now I will give you page {page_number} of the document.'''
             '''[Document Start]\n'''
             '''{document}\n\n'''
             '''[Document End]\n\n'''
             '''{format_instructions}\n''',
    input_variables=['page_number', 'meta_data_names', 'document', 'format_instructions'],
)
prompt_template = prompt_template.partial( meta_data_names=','.join(META_DATA_NAMES) )

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
    first_authors: List[str]   = Field(description = f'The first authors of the paper. The order should be preserved. {not_found_list.format()}')
    keywords: List[str]        = Field(description = f'The keywords of the paper. If there is no keywords, please extract the keywords from the abstract. If no keywords are found, please return an empty list.')
    citation_format: str       = Field(description = f'The citation format of the paper [APA format]. {not_found_NA.format()}')
    section_titles: List[str]  = Field(description = f'The section titles of the paper. {not_found_list.format()}')
    figure_captions: List[str] = Field(description = f'''The figure captions of the paper.'''
                                                     f'''The caption text associated with the figure. '''
                                                     f'''The caption text typically follows the word "Figure" or 'Fig.' and a number (e.g., Figure 1/Fig. 1) and provides a brief description or explanation of the figure.'''
                                                     f'''Do not separate the sub-images into different captions.'''
                                                     f'''Please keep figure number. {not_found_list.format()}''')
    table_captions: List[str]  = Field(description = f'The table captions of the paper. Please keep table number. {not_found_list.format()}')
    read_next_page:str         = Field(description = f'Whether next page information is needed. If needed return True, otherwise return False. {not_found_None.format()}')

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
        figure_captions = [],
        table_captions = [],
        read_next_page = 'None',
    )

DEFAULT_PARSER = PydanticOutputParser(pydantic_object = AcademicPaperMetadata)

# prompt_template_with_output_parser is the prompt template with the default output parser
prompt_template_with_output_format_instructions = prompt_template.partial(format_instructions=DEFAULT_PARSER.get_format_instructions())

def get_metadata_one_page(
        document: str, 
        chat_model:BaseChatModel,
        page_number:int, 
        prompt_template: PromptTemplate = prompt_template_with_output_format_instructions,
        output_parsor: Optional[BaseOutputParser] = DEFAULT_PARSER,
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
    
    Returns
    -------
    Any
        The metadata extracted from the document.

    """
    prompt_values = prompt_template.format(page_number=page_number, document=document)
    message = chat_model.invoke(prompt_values)
    if output_parsor:
        return output_parsor.invoke(message)
    return message

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

        # authors, affliations, emails, first_authors, keywords, section_titles, figure_captions, table_captions
        empty_list_tags = ['section_titles', 'figure_captions', 'table_captions']
        for tag in empty_list_tags:
            if getattr(data, tag):
                metadata_dict[tag] += getattr(data, tag)
        
    metadata_dict['read_next_page'] = metadata[-1].read_next_page
    return AcademicPaperMetadata(**metadata_dict)

def get_metadata(
        doucments: List[str],
        chat_model:BaseChatModel,
        prompt_template: PromptTemplate = prompt_template_with_output_format_instructions,
        output_parsor: Optional[BaseOutputParser] = DEFAULT_PARSER,
        merge: bool = True,
        auto_stop: bool = False,
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
    
    auto_stop : bool
        Whether to stop automatically when the model returns 'False'.
    
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
                                      output_parsor   = output_parsor)
        metadata.append(data)

        if auto_stop and data.read_next_page == 'False':
            break
    
    if merge:
        return merge_academic_paper_metadata(metadata)
    return metadata







