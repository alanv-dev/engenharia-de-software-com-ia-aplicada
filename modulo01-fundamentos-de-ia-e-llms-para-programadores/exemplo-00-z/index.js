import tf from '@tensorflow/tfjs';


async function trainModel(inputXs, outputYs) {
    const model = tf.sequential()

    // CAMADA DENSA OCULTA — transforma as 7 features de entrada em 80 ativações
    // calculadas. Uma feature é uma medida fornecida ao modelo: aqui, idade e
    // indicadores de cor/localização. `inputShape: [7]` precisa corresponder à
    // quantidade e à ordem usadas em todos os vetores passados ao treinamento.
    // Cada unidade (chamada informalmente de neurônio) calcula uma combinação
    // ponderada das entradas e aplica uma função de ativação.
    //
    // A camada densa calcula z = W·x + b: uma transformação afim (soma
    // ponderada das entradas mais um termo de viés b). Em seguida, `activation: 'relu'`
    // aplica f(z) = max(0, z): por exemplo, ReLU(-2)=0 e ReLU(3)=3. ReLU é
    // é linear em cada trecho, mas não no conjunto todo: a regra muda em zero.
    // Neste exemplo, ReLU é a ativação não linear da camada oculta e permite à
    // rede aprender fronteiras de decisão não lineares.
    //
    // CENÁRIO HIPOTÉTICO — se retirássemos ReLU, cada camada densa oculta faria
    // apenas uma transformação afim; empilhar essas camadas ainda equivaleria a
    // uma única transformação afim. A saída softmax continuaria produzindo scores
    // não lineares, mas não criaria fronteiras de decisão curvas: a classe de
    // maior score continuaria determinada pela comparação entre logits afins.
    // Portanto, aumentar apenas o número de camadas densas sem ReLU não daria
    // ao classificador a capacidade de aprender fronteiras não lineares.
    //
    // Há 80 unidades e só 3 exemplos de treino. A camada oculta e a saída têm
    // juntas 883 parâmetros treináveis (pesos e vieses), mas só três rótulos
    // observados. O modelo tem muito mais capacidade que evidência disponível e
    // pode memorizar esses exemplos
    // (overfitting, ou sobreajuste) sem aprender uma regra que funcione para
    // outras pessoas. Além disso, neste conjunto cada pessoa tem uma combinação
    // exclusiva de idade, cor, cidade e classe: não é possível separar qual
    // característica explica a categoria. Este desenho serve para demonstrar
    // a API, não para recomendar planos reais.
    model.add(tf.layers.dense({ inputShape: [7], units: 80, activation: 'relu' }))

    // CAMADA DE SAÍDA: há uma unidade para cada classe, na ordem
    // [premium, medium, basic]. A ordem precisa coincidir com os vetores de
    // rótulo e com `labelsNomes` abaixo; trocar uma ordem sem trocar as outras
    // faria o programa exibir o nome errado para uma saída.

    // Softmax converte os três valores brutos da saída em números positivos
    // cuja soma é 1. Ex.: [0.7, 0.2, 0.1] permite escolher `premium` como classe
    // de maior pontuação. Isso não prova que, entre casos com 0.7, exatamente
    // 70% pertençam a essa classe: para essa interpretação probabilística é
    // preciso avaliar calibração com muitos dados independentes.
    model.add(tf.layers.dense({ units: 3, activation: 'softmax' }))

    // COMPILAÇÃO escolhe o método de otimização e a medida de erro. Adam
    // (Adaptive Moment Estimation) atualiza os pesos usando estimativas da média
    // e da magnitude recente dos gradientes, ajustando as atualizações por peso.
    // É uma escolha prática comum, mas não garante encontrar a melhor solução e
    // seus resultados dependem, entre outros fatores, da taxa de aprendizado.
    //
    // `categoricalCrossentropy` (entropia cruzada categórica) mede a diferença
    // entre a distribuição prevista e o rótulo correto. Com saída softmax e
    // rótulos one-hot, `[1, 0, 0]` significa que a classe correta é `premium`.
    // Uma previsão que atribui probabilidade muito baixa à classe correta recebe
    // perda alta. Essa combinação é apropriada quando cada exemplo pertence a
    // exatamente uma classe. Se várias categorias puderem ser verdadeiras ao
    // mesmo tempo, o problema é multirrótulo e normalmente exige sigmoides por
    // classe e outra configuração de perda.

    model.compile({
        optimizer: 'adam',
        loss: 'categoricalCrossentropy',
        // Accuracy é a fração de exemplos cuja classe de maior saída está correta.
        // Com três casos, cada acerto altera a métrica em 1/3; e ela é medida
        // no treino, não em pessoas novas.
        metrics: ['accuracy']
    })

    // TREINAMENTO: em cada época (epoch), o modelo percorre os exemplos e
    // ajusta pesos para reduzir a perda. `shuffle` muda a ordem dos exemplos;
    // não cria dados novos nem corrige uma amostra enviesada.
    // Há 100 épocas e somente 3 exemplos. Repeti-los muitas vezes pode reduzir
    // a perda de treino enquanto aumenta o sobreajuste. Não há conjunto de
    // validação ou teste neste programa: a métrica de treino não mede desempenho
    // em pessoas não vistas. Em um experimento real, separar dados por pessoa
    // (e não por cópias da mesma pessoa) e comparar com uma regra simples ajuda
    // a verificar se a rede acrescenta valor.
    await model.fit(
        inputXs,
        outputYs,
        {
            verbose: 0,
            epochs: 100,
            shuffle: true,
            callbacks: {
                // onEpochEnd: (epoch, log) => console.log(
                //     `Epoch: ${epoch}: loss = ${log.loss}`
                // )
            }
        }
    )

    return model
}

async function predict(model, pessoa) {
    // Tensor é a estrutura numérica multidimensional do TensorFlow.js. A entrada
    // é 2D [exemplos, features]; mesmo uma pessoa deve ser uma linha [[...7
    // valores...]], e não um vetor 1D.
    const tfInput = tf.tensor2d(pessoa)

    // Faz a predição; com softmax, os valores são positivos e somam 1,
    // mas não são necessariamente probabilidades calibradas.
    const pred = model.predict(tfInput)
    const predArray = await pred.array()

    // `array()` copia a saída para arrays JavaScript. `dispose()` libera os
    // buffers intermediários; sem isso, predições repetidas podem acumular
    // memória no backend. O chamador libera depois os tensores de treino/modelo.
    tfInput.dispose()
    pred.dispose()

    return predArray[0].map((prob, index) => ({ prob, index }))
}
// Exemplo legível dos registros correspondentes aos vetores abaixo. Esta lista
// está comentada e não é consumida pelo modelo; as entradas efetivas são as
// matrizes numéricas posteriores.
// const pessoas = [
//     { nome: "Erick", idade: 30, cor: "azul", localizacao: "São Paulo" },
//     { nome: "Ana", idade: 25, cor: "vermelho", localizacao: "Rio" },
//     { nome: "Carlos", idade: 40, cor: "verde", localizacao: "Curitiba" }
// ];

// REPRESENTAÇÃO DA ENTRADA (feature engineering): a rede recebe números, não
// os objetos JavaScript nem os textos "azul" e "São Paulo". Por isso, as
// categorias são convertidas para one-hot encoding: um indicador por opção,
// com 1 na opção presente e 0 nas demais. Assim, azul = [1,0,0] e Rio = [0,1,0].
// A idade é contínua e foi escalada manualmente para aproximadamente [0,1].
// A ordem abaixo é um contrato: mudá-la exige mudar também cada vetor novo.
// Em produção, ajuste a escala usando apenas o conjunto de treino, reutilize
// os mesmos limites na inferência e defina o tratamento de categorias inéditas.
// const tensorPessoas = [
//     [0.33, 1, 0, 0, 1, 0, 0], // Erick
//     [0, 0, 1, 0, 0, 1, 0],    // Ana
//     [1, 0, 0, 1, 0, 0, 1]     // Carlos
// ]

// Feature é um atributo de entrada. Para categorias sem ordem natural (cor e
// cidade), one-hot evita inventar uma escala: vermelho não é maior que azul.
// A ordem dos sete valores é um contrato que deve ser mantido na inferência.
const tensorPessoasNormalizado = [
    [0.33, 1, 0, 0, 1, 0, 0], // Erick
    [0, 0, 1, 0, 0, 1, 0],    // Ana
    [1, 0, 0, 1, 0, 0, 1]     // Carlos
]

// Rótulo (label) é a resposta que queremos que o modelo aprenda. One-hot usa
// um vetor com 1 na classe correta e 0 nas demais; a posição corresponde à
// ordem [premium, medium, basic] da camada de saída.
// [premium, medium, basic]
const labelsNomes = ["premium", "medium", "basic"]; // Ordem dos labels
const tensorLabels = [
    [1, 0, 0], // premium - Erick
    [0, 1, 0], // medium - Ana
    [0, 0, 1]  // basic - Carlos
];

// xs tem shape [3,7] (três exemplos, sete features); ys tem [3,3] (três
// exemplos, três classes). Shape é a quantidade de elementos em cada eixo.
const inputXs = tf.tensor2d(tensorPessoasNormalizado)
const outputYs = tf.tensor2d(tensorLabels)

// Estes três registros são deliberadamente minúsculos e artificiais. Cada classe
// aparece uma vez, e idade/cor/cidade mudam junto com o rótulo; não há como saber
// qual atributo explica a classe. Para recomendar planos reais, seria preciso
// definir sucesso, coletar dados e rótulos confiáveis, separar treino/validação/
// teste por pessoa e comparar com regras simples. Esta rede demonstra o fluxo,
// não fornece evidência sobre preferências.
const model = await trainModel(inputXs, outputYs)

const pessoa = { nome: 'zé', idade: 28, cor: 'verde', localizacao: "Curitiba" }
// INFERÊNCIA significa usar o modelo já treinado em um exemplo novo. A idade
// deve receber exatamente a transformação usada no treino: (28 - 25)/(40 - 25)
// = 0.2. Se normalizarmos cada pessoa com seus próprios limites, o significado
// do número muda e a entrada deixa de ser comparável. Aqui os limites são
// constantes escritos à mão; um sistema real deve guardar os limites calculados
// no treino. A nova pessoa também usa uma cor e cidade conhecidas; categorias
// ausentes ou novas exigiriam uma política explícita, como uma categoria "outros".

const pessoaTensorNormalizado = [
    [
        0.2, // idade normalizada
        0,   // cor azul
        0,   // cor vermelho
        1,   // cor verde
        0,   // localização São Paulo
        0,   // localização Rio
        1    // localização Curitiba
    ]
]

const predictions = await predict(model, pessoaTensorNormalizado)
// Ordena as classes pela saída softmax e formata os valores como porcentagens.
// Isso é apenas a confiança relativa prevista pelo modelo; com três exemplos
// artificiais e sem calibração, "80%" não deve ser lido como chance real de a
// pessoa pertencer ao plano.
const results = predictions
    .sort((a, b) => b.prob - a.prob)
    .map(p => `${labelsNomes[p.index]} (${(p.prob * 100).toFixed(2)}%)`)
    .join('\n')
console.log(results)

// Libera memória dos tensores principais e do modelo
inputXs.dispose()
outputYs.dispose()
model.dispose()
