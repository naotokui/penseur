# penseur
This code provides an interface for the [original skip-thought vector code](https://github.com/ryankiros/skip-thoughts) by Ryan Kiros et al. (2015)

## Dependencies and Setup
To use the skip-thought code, you will need:
* Python 2.7
* Theano 0.7
* A recent version of [NumPy](http://www.numpy.org/) and [SciPy](http://www.scipy.org/)
* [scikit-learn](http://scikit-learn.org/stable/index.html)
* [NLTK 3](http://www.nltk.org/)
* [Keras](https://github.com/fchollet/keras) (for Semantic-Relatedness experiments only)
* [gensim](https://radimrehurek.com/gensim/) (for vocabulary expansion when training new models)

For those who haven't yet played with the original skip-thought code, it requires certain embedding files to work correctly. Details about obtaining these files are located in the "Getting Started" section of the skip-thought github page, but I've taken the opportunity to write a short download script that will run the original wget commands and place them in the proper location. The penseur code will always assume they are placed in a folder called 'data'.

```bash
chmod +x download_essential_files.sh
./download_essential_files.sh
```

**The data folder should now include the following files:** bi_skip.npz, bi_skip.npz.pkl, btable.npy, dictionary.txt, uni_skip.npz, uni_skip.npz.pkl, utable.npy

Loading an encoder model requires a word2vec .bin file (for vocabulary expansion, as discussed in the original paper). There is a link to the one [here](https://drive.google.com/file/d/0B7XkCwpI5KDYNlNUTTlSS21pQmM/edit). **Place it in the data folder.**

## Usage
For convenience, here is a [link](https://drive.google.com/open?id=0B3lpCS07rg43dml3MHVENGJoeXM) to a pickle file of a list of sentences from Larry King transcripts. It's over a million lines long and consists of transcripts of conversations from 2000-2011. I don't have enough space to host the encodings file, so you'll still have to generate that (which could take a day or so). **Place it in the data folder.**

**Any encoding models or decoders you create should be in the data folder as well, but penseur will handle that for you as long as you use the proper commands.**

During training an encoder or decoder, if Theano throws TypeError: ('An update must have the same type as the original shared variable (shared_var=\<TensorType(float32, matrix)>', etc.), adjust your python call to include specifying floatX to be equivalent to float32:

```bash
THEANO_FLAGS=mode=FAST_RUN,device=cpu,floatX=float32 python
```

The available methods are demonstrated below.

```python
import penseur

# Defaults to using the traditional skip-thought encoding model referenced in the original paper.
p = penseur.Penseur()

# Define a list of sentences
sentences = ["Where is the dog?",\
"What have you done with the cat?",\
"Why have you killed all my animals?",\
"You're a monster!",\
"Get out of my house!"]

# You can add the sentences to the vector space using the encode method
p.encode(sentences)

# You can save the encodings to a file using the save method.
# The parameter is simply a keyword for the save file
p.save('larry_king')

# Once you've saved encodings to a file, you can load them back into the model using the load method
p.load('larry_king')

# Test sentences against the vector space. This will return the sentences that most resemble the input
p.get_closest_sentences("Honey, where are my pants?")
# You can also request a specific number of results (default is 5)
p.get_closest_sentences("Honey, where are my pants?", 10)

# Test words against the vector space. This will return the words that are nearest to the query word
p.get_closest_words("dog")
# You can also request a specific number of results (default is 5)
p.get_closest_words("dog", 10)

# Use the get_vector method to return the vector for a specific sentence
vector = p.get_vector("How could you let the raptors into the building?")

# Use the get_sentence method to get the closest sentence to a vector (in the embedding space)
sentence = p.get_sentence(vector)

# Perform an analogy using pre-processed text files, defaulting to using the Larry King question set
p.analogy("Why can't every lightsaber be the same color as mine?")
# Perform an analogy using a different text file
p.analogy("Why can't every lightsaber be the same color as mine?", "different_text_filename")

# Display the sentence encodings in a 2D plot. Only works with small corpora.
p.display_PCA_plot()

# Display the sentence encodings in a 2D plot, but with axis constraints s.t. the 
# data is organized how you choose.
x_sentences = ['I have 10 cats.', 'I have 100 cats.']
y_sentences = ['You are my friend.', 'You are my enemy.']
p.display_constrained_plot(x_sentences, y_sentences)
```

The methods below are available in penseur_utils.py.

```python
# Train a new encoding model from scratch
import penseur_utils
name = 'ALPHA_data'
sentences = ["Where is the dog?",\
	"What have you done with the cat?",\
	"Why have you killed all my animals?",\
	"You're a monster!",\
	"Get out of my house!",\
	"Why are you here?",\
	"Get out of my mansion!",\
	"Get rid of my house!",\
	"Where have you put the cat?",\
	"Where is the dog with spots?"]
epochs = 6
save_frequency = 5
penseur_utils.train_encoder(name, sentences, epochs, save_frequency)

# Load an encoding model
import penseur
name = 'ALPHA_data'
p = penseur.Penseur(model_name=name)

# Train a decoder from scratch
import penseur, penseur_utils
p = penseur.Penseur()
name = 'ALPHA_data'
sentences = ["Where is the dog?",\
	"What have you done with the cat?",\
	"Why have you killed all my animals?",\
	"You're a monster!",\
	"Get out of my house!",\
	"Why are you here?",\
	"Get out of my mansion!",\
	"Get rid of my house!",\
	"Where have you put the cat?",\
	"Where is the dog with spots?"]
epochs = 6
savefreq = 5
penseur_utils.train_decoder(name, sentences, p.model, epochs, savefreq)

# Load a decoder
import penseur_utils
name = ALPHA_data
dec = penseur_utils.load_decoder(name)

# Decode a vector (returning either 1 sentence or n sentences, default is 1)
vector = p.get_vector('Where are the animals?')
just_one_sentence = penseur_utils.decode(dec, vector)
three_sentences = penseur_utils.decode(dec, vector, 3)
```

# skip-thoughts

Sent2Vec encoder and training code from the paper [Skip-Thought Vectors](http://arxiv.org/abs/1506.06726).

## Dependencies

This code is written in python. To use it you will need:

* Python 2.7
* Theano 0.7
* A recent version of [NumPy](http://www.numpy.org/) and [SciPy](http://www.scipy.org/)
* [scikit-learn](http://scikit-learn.org/stable/index.html)
* [NLTK 3](http://www.nltk.org/)
* [Keras](https://github.com/fchollet/keras) (for Semantic-Relatedness experiments only)
* [gensim](https://radimrehurek.com/gensim/) (for vocabulary expansion when training new models)

## Getting started

You will first need to download the model files and word embeddings. The embedding files (utable and btable) are quite large (>2GB) so make sure there is enough space available. The encoder vocabulary can be found in dictionary.txt.

    wget http://www.cs.toronto.edu/~rkiros/models/dictionary.txt
    wget http://www.cs.toronto.edu/~rkiros/models/utable.npy
    wget http://www.cs.toronto.edu/~rkiros/models/btable.npy
    wget http://www.cs.toronto.edu/~rkiros/models/uni_skip.npz
    wget http://www.cs.toronto.edu/~rkiros/models/uni_skip.npz.pkl
    wget http://www.cs.toronto.edu/~rkiros/models/bi_skip.npz
    wget http://www.cs.toronto.edu/~rkiros/models/bi_skip.npz.pkl

NOTE to Toronto users: You should be able to run the code as is from any machine, without having to download.

Once these are downloaded, open skipthoughts.py and set the paths to the above files (path_to_models and path_to_tables). Now you are ready to go. Make sure to set the THEANO_FLAGS device if you want to use CPU or GPU.

Open up IPython and run the following:

    import skipthoughts
    model = skipthoughts.load_model()

Now suppose you have a list of sentences X, where each entry is a string that you would like to encode. To get vectors, just run the following:

    vectors = skipthoughts.encode(model, X)

vectors is a numpy array with as many rows as the length of X, and each row is 4800 dimensional (combine-skip model, from the paper). The first 2400 dimensions is the uni-skip model, and the last 2400 is the bi-skip model. We highly recommend using the combine-skip vectors, as they are almost universally the best performing in the paper experiments.

As the vectors are being computed, it will print some numbers. The code works by extracting vectors in batches of sentences that have the same length - so the number corresponds to the current length being processed. If you want to turn this off, set verbose=False when calling encode.

The rest of the document will describe how to run the experiments from the paper. For these, create a folder called 'data' to store each of the datasets.

## TREC Question-Type Classification

Download the dataset from http://cogcomp.cs.illinois.edu/Data/QA/QC/ (train_5500.label and TREC_10.label) and put these into the data directory. To obtain the test set result using the best chosen hyperparameter from CV, run the following:

    import eval_trec
    eval_trec.evaluate(model, evalcv=False, evaltest=True)

This should give you a result of 92.2%, as in the paper. Alternatively, you can set evalcv=True to do 10-fold cross-validation on the training set. It should find the same hyperparameter and report the same accuracy as above.

## Image-Sentence Ranking

The file eval_rank.py is used for the COCO image-sentence ranking experiments. To use this, you need to prepare 3 lists: one each for training, development and testing. Each list should consist of 3 entries. The first entry is a list of sentences, the second entry is a numpy array of image features for the corresponding sentences (e.g. OxfordNet/VGG) and the third entry is a numpy array of skip-thought vectors for the corresponding sentences.

To train a model, open eval_rank.py and set the hyperparameters as desired in the trainer function. Then simply run:

    import eval_rank
    eval_rank.trainer(train, dev)

where train and dev are the lists you created. The model will train for the maximum numbers of epochs specified and periodically compute ranks on the development set. If the ranks improve, it will save the model. After training is done, you can evaluate a saved model by calling the evaluate function:

    eval_rank.evaluate(dev, saveto, evaluate=True)

This will load a saved model from the 'saveto' path and evaluate on the development set (alternatively, past the test list instead to evaluate on the test set).

Pre-computed COCO features will be made available at a later date, once I find a suitable place to host them. Note that this ranking code is generic, it can be applied with other tasks but you may need to modify the evaluation code accordingly.

## Semantic-Relatedness

Download the SemEval 2014 Task 1 (SICK) dataset from http://alt.qcri.org/semeval2014/task1/index.php?id=data-and-tools (training data, trial data and test data with annotations) and put these into the data directory. Then run the following:

    import eval_sick
    eval_sick.evaluate(model, evaltest=True)

This will train a model using the trial dataset to early stop on Pearson correlation. After stopping, it will evaluate the result on the test set. It should output the following results:

    Test Pearson: 0.858463714763
    Test Spearman: 0.791613731617
    Test MSE: 0.26871638445

For this experiment, you will need to have Keras installed in order for it to work.

## Paraphrase Detection

Download the Microsoft Research Paraphrase Corpus and put it in the data directory. There should be two files, msr_paraphrase_train.txt and msr_paraphrase_test.txt. To obtain the test set result using the best chosen hyperparameter from CV, run the following:

    import eval_msrp
    eval_msrp.evaluate(model, evalcv=False, evaltest=True, use_feats=True)

This will evaluate on the test set using the best chosen hyperparamter from CV. I get the following results:

    Test accuracy: 0.75768115942
    Test F1: 0.829526916803

Alternatively, turning on evalcv will perform 10-fold CV on the training set, and should output the same result after.

## Binary classification benchmarks

The file eval_classification.py is used for evaluation on the binary classification tasks (MR, CR, SUBJ and MPQA). You can download CR and MPQA from http://nlp.stanford.edu/~sidaw/home/projects:nbsvm and MR and SUBJ from https://www.cs.cornell.edu/people/pabo/movie-review-data/ (sentence polarity dataset, subjectivity dataset). Included is a function for nested cross-validation, since it is standard practice to report 10-fold CV on these datasets. Here is sample usage:

    import eval_classification
    eval_classification.eval_nested_kfold(model, 'SUBJ', use_nb=False)

This will apply nested CV on the SUBJ dataset without NB features. The dataset names above can be substituted in place of SUBJ.

## A note about the EOS (End-of-Sentence) token

By default the EOS token is not used when encoding, even though it was used in training. We found that this results in slightly better performance across all tasks, assuming the sentences end with proper puctuation. If this is not the case, we highly recommend using the EOS token (which can be applied with use_eos=True in the encode function). For example, the semantic-relatedness sentences have been stripped of periods, so we used the EOS token in those experiments. If ever in doubt, consider it as an extra hyperparameter.

## BookCorpus data

The pre-processed dataset we used for training our model is now available [here](http://www.cs.toronto.edu/~mbweb/).

## Reference

If you found this code useful, please cite the following paper:

Ryan Kiros, Yukun Zhu, Ruslan Salakhutdinov, Richard S. Zemel, Antonio Torralba, Raquel Urtasun, and Sanja Fidler. **"Skip-Thought Vectors."** *arXiv preprint arXiv:1506.06726 (2015).*

    @article{kiros2015skip,
      title={Skip-Thought Vectors},
      author={Kiros, Ryan and Zhu, Yukun and Salakhutdinov, Ruslan and Zemel, Richard S and Torralba, Antonio and Urtasun, Raquel and Fidler, Sanja},
      journal={arXiv preprint arXiv:1506.06726},
      year={2015}
    }

If you use the BookCorpus data in your work, please also cite:

Yukun Zhu, Ryan Kiros, Richard Zemel, Ruslan Salakhutdinov, Raquel Urtasun, Antonio Torralba, Sanja Fidler.
**"Aligning Books and Movies: Towards Story-like Visual Explanations by Watching Movies and Reading Books."** *arXiv preprint arXiv:1506.06724 (2015).*

    @article{zhu2015aligning,
        title={Aligning Books and Movies: Towards Story-like Visual Explanations by Watching Movies and Reading Books},
        author={Zhu, Yukun and Kiros, Ryan and Zemel, Richard and Salakhutdinov, Ruslan and Urtasun, Raquel and Torralba, Antonio and Fidler, Sanja},
        journal={arXiv preprint arXiv:1506.06724},
        year={2015}
    }

## License

[Apache License 2.0](http://www.apache.org/licenses/LICENSE-2.0)
