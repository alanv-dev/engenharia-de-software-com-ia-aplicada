import * as tf from '@tensorflow/tfjs';

async function trainModel(inputXs, outputYs) {
    const model = tf.sequential();

    // CAMADA DENSA OCULTA — transforma as 7 features (medidas de entrada) em
    // 80 ativações calculadas. `inputShape: [7]` deve corresponder à quantidade e
    // à ordem usadas em cada vetor de entrada.
    // A camada densa calcula z = W·x + b: uma transformação afim (soma ponderada
    // das entradas mais um termo de viés b). ReLU (Rectified Linear Unit) aplica
    // f(z) = max(0,z): por exemplo, ReLU(-2)=0 e ReLU(3)=3. É linear em cada
    // trecho, mas não no conjunto todo, porque sua regra muda em zero. Neste
    // modelo, ReLU permite à camada oculta aprender fronteiras de decisão não
    // lineares.
    //
    // CENÁRIO HIPOTÉTICO — se retirássemos ReLU, empilhar camadas densas ocultas
    // equivaleria a uma única transformação afim; mais camadas, sozinhas, não
    // criariam fronteiras de decisão não lineares. A softmax da saída ainda
    // transformaria os logits afins em scores não lineares, mas não mudaria qual
    // logit é maior, portanto não tornaria curvas as fronteiras entre classes.
    //
    // As 80 unidades (chamadas informalmente de neurônios) são uma escolha para
    // demonstrar a API, não uma escolha baseada em experimento. Existem só três
    // exemplos de treino: a rede pode memorizá-los (overfitting/sobreajuste) e
    // falhar em pessoas novas. Idade, cor, cidade e classe também mudam juntas
    // nesses três registros, então não há evidência para descobrir qual feature
    // explica a classe. Para uma aplicação real, colete dados representativos,
    // separe validação/teste por pessoa e compare com modelos mais simples.
    model.add(tf.layers.dense({ inputShape: [7], units: 80, activation: 'relu' }));

    // CAMADA DE SAÍDA: há uma unidade para cada classe, na ordem [premium,
    // medium, basic]. Essa ordem deve coincidir com os rótulos de treino e com
    // `labelsNomes`; caso contrário, o número pode ser associado ao nome errado.
    // Softmax transforma os valores brutos em números positivos cuja soma é 1.
    // Ex.: [0.7, 0.2, 0.1] escolhe premium como classe de maior pontuação, mas
    // não garante que 70% dos casos com esse resultado sejam premium. Essa
    // interpretação requer avaliar calibração com dados independentes.
    model.add(tf.layers.dense({ units: 3, activation: 'softmax' }));

    // COMPILAÇÃO define o otimizador (como os pesos serão atualizados) e a perda
    // (o erro que queremos reduzir). Adam (Adaptive Moment Estimation) usa médias
    // recentes da direção e magnitude dos gradientes para adaptar as atualizações
    // de cada peso. É uma opção comum, não uma garantia de solução ótima.
    // `categoricalCrossentropy` (entropia cruzada categórica) compara a saída com
    // o rótulo one-hot: [1,0,0] quer dizer que premium é a classe correta. Uma
    // previsão que dá pouca pontuação à classe correta recebe perda alta. Esta
    // combinação serve quando cada exemplo tem exatamente uma classe. Se várias
    // classes puderem ser verdadeiras simultaneamente, use formulação multirrótulo
    // (por exemplo, sigmoide independente por classe e perda correspondente).
    model.compile({
        optimizer: 'adam',
        loss: 'categoricalCrossentropy',
        // Accuracy é a fração de exemplos cuja classe de maior saída está correta.
        // Com apenas três casos, cada acerto muda a métrica em 1/3; ela também
        // é medida no treino, então não estima o desempenho em pessoas novas.
        metrics: ['accuracy']
    });

    // TREINAMENTO ajusta os pesos para reduzir a perda nos exemplos fornecidos.
    // Uma época (epoch) é uma passagem pelo conjunto de treino; 100 épocas com
    // apenas três exemplos repetem a mesma evidência muitas vezes e favorecem
    // sobreajuste. Não há validação neste exemplo, portanto a perda/accuracy do
    // treino não diz como o modelo se comporta em pessoas novas. `shuffle` altera
    // a ordem das amostras e pode reduzir efeitos da ordem, mas não corrige viés,
    // rótulos errados nem falta de diversidade.
    await model.fit(
        inputXs,
        outputYs,
        {
            verbose: 0,     // 0 silencia os logs; habilite a saída para acompanhar o ajuste
            epochs: 100,    // 100 passagens completas pelo dataset de treino
            shuffle: true   // Embaralha as amostras; isso reduz efeitos da ordem, mas não remove vieses dos dados
        }
    );

    return model;
}

async function predict(model, pessoa) {
    // Tensor é a estrutura numérica multidimensional do TensorFlow.js. A entrada
    // tem formato [exemplos, features]; mesmo uma pessoa ocupa uma linha com
    // sete valores, por isso `pessoa` é passado como [[...7 valores...]].
    const tfInput = tf.tensor2d(pessoa);

    // Faz a predição; com softmax, os valores de saída são positivos e somam 1.
    // Interprete-os como probabilidades do modelo, não necessariamente calibradas.
    const pred = model.predict(tfInput);
    const predArray = await pred.array();

    // Libera a memória alocada pelos tensores intermediários
    tfInput.dispose();
    pred.dispose();

    return predArray[0].map((prob, index) => ({ prob, index }));
}

// Exemplo ilustrativo dos registros que correspondem aos vetores abaixo:
// const pessoas = [
//     { nome: "Erick", idade: 30, cor: "azul", localizacao: "São Paulo" },
//     { nome: "Ana", idade: 25, cor: "vermelho", localizacao: "Rio" },
//     { nome: "Carlos", idade: 40, cor: "verde", localizacao: "Curitiba" }
// ];

// REPRESENTAÇÃO DAS ENTRADAS: uma feature é uma medida usada pelo modelo.
// A rede recebe números, então convertemos cor e cidade com one-hot encoding:
// cada categoria tem uma posição e a categoria presente recebe 1 (as outras,
// 0). A ordem é [idade, azul, vermelho, verde, São Paulo, Rio, Curitiba] e deve
// permanecer idêntica em treino e inferência.
// A idade usa min-max: (idade - mínimo)/(máximo - mínimo). Aqui, 25 anos vira
// 0 e 40 vira 1. Uma idade fora dessa faixa gera valor menor que 0 ou maior que
// 1; isso é possível, mas requer decisão consciente. Em aplicações reais,
// calcule limites só no treino, persista-os e defina como tratar novas categorias.
const tensorPessoasNormalizado = [
    [0.3333, 1, 0, 0, 1, 0, 0], // Erick (30 anos, azul, São Paulo)
    [0,    0, 1, 0, 0, 1, 0], // Ana   (25 anos, vermelho, Rio)
    [1,    0, 0, 1, 0, 0, 1]  // Carlos(40 anos, verde, Curitiba)
];

// Rótulos (labels) são as respostas esperadas durante o treino. One-hot usa um
// vetor com 1 na classe correta e 0 nas demais; sua ordem coincide com a saída.
// Ordem: [premium, medium, basic]
const labelsNomes = ["premium", "medium", "basic"];
const tensorLabels = [
    [1, 0, 0], // premium - Erick
    [0, 1, 0], // medium  - Ana
    [0, 0, 1]  // basic   - Carlos
];

// `xs` são as entradas; `ys`, os rótulos desejados. Tensor é a estrutura numérica
// multidimensional que o TensorFlow.js usa para processar os exemplos em lote.
const inputXs = tf.tensor2d(tensorPessoasNormalizado);
const outputYs = tf.tensor2d(tensorLabels);

// Treinamento do modelo
const model = await trainModel(inputXs, outputYs);

// INFERÊNCIA: aplicação do modelo treinado a um exemplo novo. A pessoa abaixo
// tem cor e cidade já vistas; uma categoria desconhecida não tem posição neste
// vetor e exigiria uma política (por exemplo, reservar a posição "outros").
const pessoa = { nome: 'zé', idade: 28, cor: 'verde', localizacao: "Curitiba" };

// Reutilizamos os mesmos limites de treino: (28 - 25)/(40 - 25) = 0.2. Não
// recalculamos mínimo/máximo para cada entrada, pois isso mudaria a escala que
// o modelo aprendeu. Neste exemplo, limites e codificação estão escritos à mão;
// um sistema real deve persistir o pré-processamento junto do modelo.
// Codificação one-hot respeitando a ordem:
// [idade_normalizada, azul, vermelho, verde, São Paulo, Rio, Curitiba]
const pessoaTensorNormalizado = [
    [
        0.2, // idade normalizada (28 anos)
        0,   // cor: azul
        0,   // cor: vermelho
        1,   // cor: verde
        0,   // localização: São Paulo
        0,   // localização: Rio
        1    // localização: Curitiba
    ]
];

const predictions = await predict(model, pessoaTensorNormalizado);
// A ordenação seleciona a classe de maior saída. A porcentagem é a saída
// softmax formatada, não uma estimativa validada da chance real: o modelo viu
// apenas três exemplos e não passou por teste de calibração.
const results = predictions
    .sort((a, b) => b.prob - a.prob)
    .map(p => `${labelsNomes[p.index]} (${(p.prob * 100).toFixed(2)}%)`)
    .join('\n');

console.log("Resultado da classificação para Zé:");
console.log(results);

// Libera memória dos tensores principais e do modelo
inputXs.dispose();
outputYs.dispose();
model.dispose();
