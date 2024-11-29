# AttrAlign
This is code and datasets for AttrAlign

## Dependencies

1. Python 3.9
2. PyTorch 2.3.0
3. Numpy
4. Llama 3
## Installation

Install llama3 according to https://github.com/meta-llama/llama3. 
## Dataset
DBP15K and OpenEA

All the data can be downloaded from https://drive.google.com/file/d/1pB6vcJs2WrBTXxXbZ582nBzrDWV4oCa5/view?usp=drive_link.

Initial DBP15K datasets are from JAPE(https://github.com/nju-websoft/JAPE). Initial OpenEA datasets are from https://github.com/nju-websoft/OpenEA?tab=readme-ov-file.

## How to Run

1. Run without LLM.
   '''
   python process_attr.py #get the attribute files
   python attr.py/attr_n.py/attr_v.py #get the alignment results with attribute information
   python combine_value.py #get the final result
   '''
2. Run with LLM
   Download code from Seg_Align (https://github.com/ yangxiaoxiaoly/Seg-Align.).
   Install llama3
   '''
   python rerank_llama.py #get the rerank results
   python get_llama_result.py #get the final results
   '''

   
   



   







