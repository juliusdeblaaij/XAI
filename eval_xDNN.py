import numpy as np

from xDNN_run_custom import train, classify
from doc2vec import doc2vec
from myutils import *

training_results = train()
prototypes =  training_results["xDNNParms"]["Parameters"]
training_parameters = training_results["xDNNParms"]

"""
test_results = validate(training_parameters)
estimated_labels = test_results["EstLabs"]
"""

case = "as a performance tester id like to investigate why theres high cpu startup time for both admin and container servers perhaps profiling would assist isolating the bottlenecks scope identify the bottlenecks document reasons list proscons"
# case = "when using --deletefile option for filejdbc all files are deleted before processing complete in this scenario i copied 2 files fakefilecsv anotherfilecsv into the bar directory created thew following job job create myjob2 --definition filejdbc --resourcesfilebarcsv --namesname --tablenamepeople2 --initializedatabasetrue --commitinterval1 --deletefilestrue --deploy then launched the job job launch myjob2 the result was that the first file was processed but then the module deleted all csv files in the directory before processing the 2nd file the net result was that the job failed and the 2nd file was not processed"
# case = "As a Spring XD on CF user I'd like to use {{cloudController}} implementation of Admin SPI every time I deploy Spring XD modules so I can leverage the SPI to query for module status and health metrics.  *Possible APIs:* {code}  ModuleStatus getStatus(ModuleDescriptor descriptor)  Collection<ModuleDescriptor> listModules()  Map<ModuleDescriptor.Key ModuleStatus>  {code}"
# case = "as a spring xd developer id like to have a permanent location of spi implementations so i could use the common repo every time i contribute or enhance the test coverage"
cleaned_text = pre_process_text(case)

case_embedding = doc2vec(cleaned_text, 'd2v_23k_dbow.model')
case_embedding = np.array(case_embedding)
features = np.array([case_embedding])
classification_result = classify(training_results, features)
estimated_labels = classification_result["EstLabs"]

for label in estimated_labels:
    prototype = prototypes[int(label[0])]
    visual_prototype = prototype["Prototype"][1]

    story_point = 0

    label = int(label[0])
    if label == 0:
        story_point = 0
    elif label == 1:
        story_point = 1
    elif label == 2:
        story_point = 2
    elif label == 3:
        story_point = 3
    elif label == 4:
        story_point = 5
    elif label == 5:
        story_point = 8

    print(case + ":")
    print(f'Story points: {story_point} {visual_prototype}')
