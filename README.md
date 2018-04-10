# RNNSharp
RNNSharp - это инструментарий глубокой рекуррентной нейронной сети, который широко используется для множества различных задач, таких как маркировка последовательности, seq2seq и так далее. Он написан на языке C# и основан на .NET framework 4.6 или выше.

На этой странице представлено, что такое RNNSharp, как он работает и как его использовать. Чтобы получить демонстрационный пакет, вы можете обратиться к странице выпуска

## Overview
RNNSharp поддерживает множество различных типов структур глубокой рекуррентной нейронной сети (aka DeepRNN).

Для структуры сети он поддерживает прямой RNN и двунаправленный RNN. Вперед RNN рассматривает историческую информацию до текущего токена, однако двунаправленный RNN рассматривает как информацию о предыстории, так и информацию в будущем.

Для скрытой структуры слоев он поддерживает LSTM и Dropout. По сравнению с BPTT, LSTM очень хорош в сохранении долговременной памяти, так как у нее есть некоторые ворота для потоковой передачи информации. Выпадение используется для добавления шума во время обучения, чтобы избежать переобучения.

С точки зрения структуры выходного уровня поддерживаются простые, softmax, выборочные softmax и повторяющиеся CRF [1]. Softmax - это тип tranditional, который широко используется во многих видах задач. Сэмплированный softmax особенно используется для задач с большим выходным лексиконом, таких как задачи генерации последовательности (модель последовательности к последовательности). Простой тип обычно используется с повторяющимся CRF вместе. Для повторного CRF, основанного на простых переходах выходов и тегов, он вычисляет вывод CRF для всей последовательности. Для задач маркировки последовательностей в автономном режиме, таких как сегментация слов, распознавание имен объектов и т. Д., Рекуррентный CRF имеет лучшую производительность, чем softmax, дискретный softmax и линейный CRF.  


Вот пример глубокой двунаправленной сети RNN-CRF. Он содержит 3 скрытых слоя, 1 собственный RNN-выход и 1 CRF-выход.
![](https://github.com/zhongkaifu/RNNSharp/blob/master/RNNSharpOverview.jpg)


Вот внутренняя структура одного двунаправленного скрытого слоя.  
![](https://github.com/zhongkaifu/RNNSharp/blob/master/RNNSharpLayer.jpg)

Вот нейронная сеть для задачи seq2seq. «TokenN» - из исходной последовательности, а «ELayerX-Y» - скрытые слои автокодера. Автокодер определяется в файле конфигурации функции. & Lt; & s GT; всегда является началом целевого предложения, а «DLayerX-Y» означает скрытые слои декодера. В декодере он генерирует один токен за один раз, пока &lt;/s&gt; генерируется.  
![](https://github.com/zhongkaifu/RNNSharp/blob/master/RNNSharpSeq2Seq.jpg)

## Supported Feature Types

RNNSharp поддерживает множество различных типов объектов, поэтому в следующем параграфе будет показано, как работают эти функции. 

## Template Features

Шаблонные функции создаются с помощью шаблонов. Благодаря заданным шаблонам и корпусу эти функции могут автоматически генерироваться. В RNNSharp функции шаблона - редкие функции, поэтому, если функция существует в текущем токене, значение функции будет равно 1 (или частоте функции), в противном случае это будет 0. Это похоже на функции CRFSharp. В RNNSharp TFeatureBin.exe является консольным инструментом для создания этого типа функций.

В файле шаблона каждая строка описывает один шаблон, который состоит из префикса, id и строки правил. Префикс указывает тип шаблона. Пока что RNNSharp поддерживает функцию U-типа, поэтому префикс всегда равен «U». Идентификатор используется для различения разных шаблонов. Правило-строка - это тело объекта.

\# Unigram  
U01:%x[-1,0]  
U02:%x[0,0]  
U03:%x[1,0]  
U04:%x[-1,0]/%x[0,0]   
U05:%x[0,0]/%x[1,0]  
U06:%x[-1,0]/%x[1,0]  
U07:%x[-1,1]  
U08:%x[0,1]  
U09:%x[1,1]  
U10:%x[-1,1]/%x[0,1]  
U11:%x[0,1]/%x[1,1]  
U12:%x[-1,1]/%x[1,1]  
U13:C%x[-1,0]/%x[-1,1]   
U14:C%x[0,0]/%x[0,1]  
U15:C%x[1,0]/%x[1,1]  


Строка правил имеет два типа: одна - постоянная строка, а другая - переменная. Простейший переменный формат {“%x[row,col]”}. Строка определяет смещение между текущим токеном фокусировки и генерирует токен функции в строке. Col указывает абсолютную позицию столбца в корпусе. Более того, переменная комбинация также поддерживается, например: {“%x[row1, col1]/%x[row2, col2]”}. Когда мы создадим набор функций, переменная будет расширена до определенной строки. Ниже приведен пример обучения данных для задачи с именем entity. 

Word       | Pos  | Tag
-----------|------|----
!          | PUN  | S
Tokyo      | NNP  | S_LOCATION
and        | CC   | S
New        | NNP  | B_LOCATION
York       | NNP  | E_LOCATION
are        | VBP  | S
major      | JJ   | S
financial  | JJ   | S
centers    | NNS  | S
.          | PUN  | S
---empty line---
!          | PUN  | S
p          | FW   | S
'          | PUN  | S
y          | NN   | S
h          | FW   | S
44         | CD   | S
University | NNP  | B_ORGANIZATION
of         | IN   | M_ORGANIZATION
Texas      | NNP  | M_ORGANIZATION
Austin     | NNP  | E_ORGANIZATION

Согласно вышеприведенным шаблонам, если текущий токен фокусировки - «York NNP E_LOCATION», ниже генерируются функции:  

U01:New  
U02:York  
U03:are  
U04:New/York  
U05:York/are  
U06:New/are  
U07:NNP  
U08:NNP  
U09:are  
U10:NNP/NNP  
U11:NNP/VBP  
U12:NNP/VBP  
U13:CNew/NNP  
U14:CYork/NNP  
U15:Care/VBP  

Хотя строки правил U07 и U08, U11 и U12 одинаковы, мы все равно можем различать их по строке id.

## Context Template Features

Функции контекстного шаблона основаны на шаблонных функциях и объединены с контекстом. В этом примере, если параметр контекста равен «-1,0,1», эта функция объединит функции текущего токена с его предыдущим токеном и следующим токеном. Например, если предложение «как вы». сгенерированный набор функций будет {Feature ("how"), Feature ("are"), Feature ("you")}.

## Pretrained Features

RNNSharp поддерживает два типа предварительно обработанных функций. Один из них - встроенные функции, а другой - функции автоматического кодирования. Оба они могут представить данный токен вектором фиксированной длины. Эта функция является плотной функцией в RNNSharp.

Для внедрения функций они обучаются из несвязанного корпуса проектом Text2Vec. И RNNSharp использует их как статические функции для каждого заданного токена. Однако для функций автоматического кодирования они также обучаются RNNSharp, а затем они могут использоваться как плотные функции для других тренировок. Обратите внимание, что гранулярность маркера в предварительно обработанных элементах должна соответствовать учебному корпусу в основном обучении, в противном случае некоторые жетоны будут неправильно совпадать с предварительной процедурой.

Любит шаблоны, функция встраивания также поддерживает функцию контекста. Он может объединить все функции заданных контекстов в единую функцию внедрения. Для функций автоматического кодирования он пока не поддерживается.

## Run Time Features

По сравнению с другими функциями, созданными в автономном режиме, эта функция генерируется во время выполнения. Он использует результат предыдущих токенов в качестве функции времени выполнения для текущего токена. Эта функция доступна только для прямого RNN, двунаправленная RNN не поддерживает ее.

## Source Sequence Encoding Feature

Эта функция предназначена только для последовательности к последовательности. В задаче последовательности-последовательности RNNSharp кодирует заданную исходную последовательность в вектор фиксированной длины и затем передает ее как плотную функцию для генерации целевой последовательности. 

## Configuration File

Файл конфигурации описывает структуру и функции модели. В консольном инструменте используйте параметр -cfgfile в качестве параметра для указания этого файла. Ниже приведен пример задачи по маркировке последовательностей:  

\#Working directory. It is the parent directory of below relatived paths.  
CURRENT_DIRECTORY = .  
  
\#Network type. Four types are supported:  
\#For sequence labeling tasks, we could use: Forward, BiDirectional, BiDirectionalAverage  
\#For sequence-to-sequence tasks, we could use: ForwardSeq2Seq  
\#BiDirectional type concatnates outputs of forward layer and backward layer as final output  
\#BiDirectionalAverage type averages outputs of forward layer and backward layer as final output  
NETWORK_TYPE = BiDirectional  
  
\#Model file path  
MODEL_FILEPATH = Data\Models\ParseORG_CHS\model.bin  
  
\#Hidden layers settings. LSTM and Dropout are supported. Here are examples of these layer types.  
\#Dropout: Dropout:0.5 -- Drop out ratio is 0.5 and layer size is the same as previous layer.  
\#If the model has more than one hidden layer, each layer settings are separated by comma. For example:  
\#"LSTM:300, LSTM:200" means the model has two LSTM layers. The first layer size is 300, and the second layer size is 200.  
HIDDEN_LAYER = LSTM:200  
  
\#Output layer settings. Simple, Softmax ands sampled softmax are supported. Here is an example of sampled softmax:  
\#"SampledSoftmax:20" means the output layer is sampled softmax layer and its negative sample size is 20.  
\#"Simple" means the output is raw result from output layer. "Softmax" means the result is based on "Simple" result and run softmax.  
OUTPUT_LAYER = Simple  
  
\#CRF layer settings  
\#If this option is true, output layer type must be "Simple" type.  
CRF_LAYER = True  
  
\#The file name for template feature set  
TFEATURE_FILENAME = Data\Models\ParseORG_CHS\tfeatures  
\#The context range for template feature set. In below, the context is current token, next token and next after next token  
TFEATURE_CONTEXT = 0,1,2  
\#The feature weight type. Binary and Freq are supported  
TFEATURE_WEIGHT_TYPE = Binary  
  
\#Pretrained features type: 'Embedding' and 'Autoencoder' are supported.  
\#For 'Embedding', the pretrained model is trained by Text2Vec, which looks like word embedding model.  
\#For 'Autoencoder', the pretrained model is trained by RNNSharp itself.  For sequence-to-sequence task, "Autoencoder" is required, since source sequence needs to be encoded by this model at first, and then target sequence would be generated by decoder.  
PRETRAIN_TYPE = Embedding  
  
\#The following settings are for pretrained model in 'Embedding' type.  
\#The embedding model generated by Txt2Vec (https://github.com/zhongkaifu/Txt2Vec). If it is raw text format, we should use WORDEMBEDDING_RAW_FILENAME instead of WORDEMBEDDING_FILENAME as keyword  
WORDEMBEDDING_FILENAME = Data\WordEmbedding\wordvec_chs.bin  
\#The context range of word embedding. In below example, the context is current token, previous token and next token  
\#If more than one token are combined, this feature would use a plenty of memory.  
WORDEMBEDDING_CONTEXT = -1,0,1  
\#The column index applied word embedding feature  
WORDEMBEDDING_COLUMN = 0  
  
\#The following setting is for pretrained model in 'Autoencoder' type.  
\#The feature configuration file for pretrained model.  
AUTOENCODER_CONFIG = D:\RNNSharpDemoPackage\config_autoencoder.txt  
  
\#The following setting is the configuration file for source sequence encoder which is only for sequence-to-sequence task that MODEL_TYPE equals to SEQ2SEQ.  
\#In this example, since MODEL_TYPE is SEQLABEL, so we comment it out.  
\#SEQ2SEQ_AUTOENCODER_CONFIG = D:\RNNSharpDemoPackage\config_seq2seq_autoencoder.txt  
  
\#The context range of run time feature. In below example, RNNSharp will use the output of previous token as run time feature for current token  
\#Note that, bi-directional model does not support run time feature, so we comment it out.  
\#RTFEATURE_CONTEXT = -1
  
## Training file format

В обучающем файле каждая последовательность представляется как матрица признаков и заканчивается пустой строкой. В матрице каждая строка предназначена для одного символа последовательности и ее функций, а каждый столбец предназначен для одного типа объектов. Во всем учебном корпусе число столбцов должно быть фиксированным.

Задача маркировки последовательности и последовательность для последовательности задают разные форматы учебного корпуса. 

### Sequence labeling corpus  

Для задач маркировки последовательностей первые столбцы N-1 являются входными функциями для обучения, а N-й столбец (последний столбец) является ответом текущего токена. Вот пример для задачи распознавания имен по имени (полный учебный файл находится в разделе выпуска, вы можете скачать его там): 

Word       | Pos  | Tag
-----------|------|----
!          | PUN  | S
Tokyo      | NNP  | S_LOCATION
and        | CC   | S
New        | NNP  | B_LOCATION
York       | NNP  | E_LOCATION
are        | VBP  | S
major      | JJ   | S
financial  | JJ   | S
centers    | NNS  | S
.          | PUN  | S
---empty line---
!          | PUN  | S
p          | FW   | S
'          | PUN  | S
y          | NN   | S
h          | FW   | S
44         | CD   | S
University | NNP  | B_ORGANIZATION
of         | IN   | M_ORGANIZATION
Texas      | NNP  | M_ORGANIZATION
Austin     | NNP  | E_ORGANIZATION


Он имеет две записи, разделенные полосой. Для каждого токена он имеет три столбца. Первые два столбца - это набор функций ввода, которые являются словом и позиционным тегом для токена. Третий столбец является идеальным выходом модели, которая называется типом сущности для токена.

Именованный тип объекта выглядит как «Position_NamedEntityType». «Позиция» - это позиция слова в названном объекте, а «NamedEntityType» - это тип объекта. Если «NamedEntityType» пуст, это означает, что это общее слово, а не именованный объект. В этом примере «Позиция» имеет четыре значения:  
 S : the single word of the named entity  
 B : the first word of the named entity  
 M : the word is in the middle of the named entity  
 E : the last word of the named entity  

"NamedEntityType" has two values:  
 ORGANIZATION : the name of one organization  
 LOCATION : the name of one location  

### Sequence-to-sequence corpus  

Для задачи seq2seq формат учебного корпуса отличается. Для каждой пары последовательностей она состоит из двух секций, одна из которых является исходной, другая - целевой. Вот пример:  
 
Word      |  
----------| 
What      | 
is        | 
your      | 
name      | 
?         | 
---empty line---
I         | 
am        | 
Zhongkai  | 
Fu        | 

В приведенном выше примере: «Как тебя зовут?» является исходным предложением, а «Я - Zhongkai Fu» является целевым предложением, созданным моделью Seq-to-seq RNNSharp. В исходном предложении, помимо функций слова, другие функции могут также применяться для обучения, такие как функция postag в задаче последовательности, указанная выше. 


## Test file format

Тестовый файл имеет такой же формат, как и файл тренировки. Для задачи маркировки последовательностей, единственная разница между ними - последний столбец. В тестовом файле все столбцы являются функциями для декодирования модели. Для задачи последовательности к последовательности она содержит только исходную последовательность. Целевое предложение будет сгенерировано моделью.  

## Tag (Output Vocabulary) File

Для задачи маркировки последовательностей этот файл содержит набор выходных тегов. Для задачи последовательности к последовательности это выходной лексический файл. 

## Console Tool

### RNNSharpConsole

RNNSharpConsole.exe - консольный инструмент для повторной кодировки и декодирования нейронной сети. Инструмент имеет два режима работы. Режим «train» предназначен для обучения модели, а режим «test» предназначен для прогнозирования выходного тега из тестового корпуса по заданной кодированной модели.


### Encode Model


В этом режиме консольный инструмент может кодировать модель RNN с помощью заданного набора функций и обучения / проверенного корпуса. Использование:

RNNSharpConsole.exe -mode train <параметры> 
 Parameters for training RNN based model.
-trainfile <string>: Training corpus file  
-validfile <string>: Validated corpus for training  
-cfgfile <string>: Configuration file  
-tagfile <string>: Output tag or vocabulary file  
-inctrain <boolean>: Incremental training. Starting from output model specified in configuration file. Default is false  
-alpha <float>: Learning rate, Default is 0.1  
-maxiter <int>: Maximum iteration for training. 0 is no limition, Default is 20  
-savestep <int>: Save temporary model after every <int> sentence, Default is 0  
-vq <int> : Model vector quantization, 0 is disable, 1 is enable. Default is 0  
-minibatch <int> : Updating weights every <int> sequence. Default is 1

Example: RNNSharpConsole.exe -mode train -trainfile train.txt -validfile valid.txt -cfgfile config.txt -tagfile tags.txt -alpha 0.1 -maxiter 20 -savestep 200K -vq 0 -grad 15.0  -minibatch 128

### Decode Model

В этом режиме, при заданном тестовом корпусе, RNNSharp предсказывает выходные теги в задаче последовательности пометок или генерирует целевую последовательность в последовательности-последовательности.
  

RNNSharpConsole.exe -mode test <parameters>  
 Parameters for predicting iTagId tag from given corpus  
-testfile <string>: test corpus file  
-tagfile <string>: output tag or vocabulary file  
-cfgfile <string>: configuration file  
-outfile <string>: result output file  

Example: RNNSharpConsole.exe -mode test -testfile test.txt -tagfile tags.txt -cfgfile config.txt -outfile result.txt    

## TFeatureBin

Он используется для создания набора шаблонов, заданных с помощью данного шаблона и файлов корпусов. Для высокопроизводительного доступа и экономии памяти индексный набор функций встроен в массив float в trie-tree от AdvUtils. Инструмент поддерживает три режима следующим образом:


TFeatureBin.exe <parameters>  
 The tool is to generate template feature from corpus and index them into file  
-mode <string> : support extract,index and build modes  
   extract : extract features from corpus and save them as raw text feature list  
   index : build indexed feature set from raw text feature list  
   build : extract features from corpus and generate indexed feature set  

### Build mode

Этот режим предназначен для извлечения функций из данного корпуса в соответствии с шаблонами, а затем для создания индексированного набора функций. Использование этого режима следующим образом:

TFeatureBin.exe -mode build <parameters>  
 This mode is to extract feature from corpus and generate indexed feature set  
-template <string> : feature template file  
-inputfile <string> : file used to generate features  
-ftrfile <string> : generated indexed feature file  
-minfreq <int> : min-frequency of feature  

Example: TFeatureBin.exe -mode build -template template.txt -inputfile train.txt -ftrfile tfeature -minfreq 3  

In above example, feature set is extracted from train.txt and build them into tfeature file as indexed feature set.  

### Extract mode

Этот режим предназначен только для извлечения функций из данного корпуса и сохранения их в исходный текстовый файл. Разница между режимом сборки и режимом извлечения заключается в том, что функция сборки режима экстракции устанавливается как формат необработанного текста, а не индексированный двоичный формат. Использование режима извлечения следующим образом:

TFeatureBin.exe -mode extract <parameters>  
 This mode is to extract features from corpus and save them as text feature list  
-template <string> : feature template file  
-inputfile <string> : file used to generate features  
-ftrfile <string> : generated feature list file in raw text format  
-minfreq <int> : min-frequency of feature  

Example: TFeatureBin.exe -mode extract -template template.txt -inputfile train.txt -ftrfile features.txt -minfreq 3  

В приведенном выше примере, в соответствии с шаблонами, набор функций извлекается из train.txt и сохраняет их в feature.txt как формат необработанного текста. Формат выходного текстового файла - «строка функции \t частота в корпусе». Вот несколько примеров:

U01:仲恺 \t 123  
U01:仲文 \t 10  
U01:仲秋 \t 12  

U01:仲恺 is feature string and 123 is the frequency that this feature in corpus.  

### Index mode

Этот режим предназначен только для создания индексированной функции, заданной заданными шаблонами и набором функций в формате необработанного текста. Использование этого режима следующим образом:

TFeatureBin.exe -mode index <parameters>  
 This mode is to build indexed feature set from raw text feature list  
-template <string> : feature template file  
-inputfile <string> : feature list in raw text format  
-ftrfile <string> : indexed feature set  

Example: TFeatureBin.exe -mode index -template template.txt -inputfile features.txt -ftrfile features.bin  

В приведенном выше примере, согласно шаблонам, набор функций сырого текста, feature.txt, будет индексироваться как файл features.bin в двоичном формате.

## Performance
Вот качественные результаты по китайской задаче распознавания сущностей. Файлы корпуса, конфигурации и параметров доступны в файле демонстрационного пакета RNNSharp в разделе [release] (https://github.com/zhongkaifu/RNNSharp/releases). Результат основан на двунаправленной модели. Первый скрытый размер слоя - 200, а второй размер скрытого слоя - 100. Вот результаты теста:

Parameter               | Token Error  | Sentence Error
------------------------|--------------|----
1-hidden layer          | 5.53%        | 15.46%
1-hidden layer-CRF      | 5.51%        | 13.60%
2-hidden layers         | 5.47%        | 14.23%
2-hidden layers-CRF     | 5.40%        | 12.93%

## Run on Linux/Mac

RNNSharp - это чистый проект C #, поэтому он может быть скомпилирован .NET Core и Mono и без изменений в Linux / Mac.

## APIs

RNNSharp также предоставляет некоторые API для разработчиков, чтобы использовать его в своих проектах. Загрузив исходный пакет кода и откройте проект RNNSharpConsole, вы увидите, как использовать API в своем проекте для кодирования и декодирования моделей RNN. Обратите внимание, что перед использованием API RNNSharp вы должны добавить RNNSharp.dll в качестве ссылки в свой проект.

## RNNSharp referenced by the following published papers  
1. [Project-Team IntuiDoc: Intuitive user interaction for document](https://www.irisa.fr/intuidoc/data/ra/intuidoc2015.pdf)
2. [A New Pre-training Method for Training Deep Learning Models with Application to Spoken Language Understanding](https://www.microsoft.com/en-us/research/wp-content/uploads/2016/07/IS2016.CameraReady-1.pdf)
3. [Long Short-Term Memory](http://pages.cs.wisc.edu/~shavlik/cs638/lectureNotes/Long%20Short-Term%20Memory%20Networks.pdf)
4. [Deep Learning](http://emma.memect.com/t/1e8f2b393ae2526a3c303ca6f3946e158f518d3a0ad6f5287967a2229395e9a2/Deep%20Learning.pdf)
