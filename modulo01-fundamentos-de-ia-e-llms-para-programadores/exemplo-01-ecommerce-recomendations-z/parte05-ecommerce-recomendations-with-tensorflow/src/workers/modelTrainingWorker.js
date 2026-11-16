import 'https://cdn.jsdelivr.net/npm/@tensorflow/tfjs@4.22.0/dist/tf.min.js';
import { workerEvents } from '../events/constants.js';
let _globalCtx = {};
let _model = null

// Pesos manuais para os grupos de atributos. Eles somam 1, mas isso não os
// transforma em probabilidades nem prova que categoria (0.4) deva importar
// quatro vezes mais que idade (0.1). São uma decisão de demonstração, aplicada
// antes do treino; escolha-os com hipótese explícita e avaliação, ou deixe o
// modelo aprender escalas a partir de dados suficientes. Features correlacionadas
// ou repetidas podem contar duas vezes mesmo quando os pesos parecem equilibrados.
const WEIGHTS = {
    category: 0.4,
    color: 0.3,
    price: 0.2,
    age: 0.1,
};


// NORMALIZAÇÃO min-max: (valor - mínimo)/(máximo - mínimo). Se preços forem
// 40 e 200, por exemplo, 120 vira (120-40)/(200-40)=0.5. A escala pode ajudar
// a otimização quando atributos têm unidades distintas, mas não lhes dá a mesma
// importância semântica. Valores futuros fora dos limites podem ficar abaixo de
// 0 ou acima de 1. O fallback para denominador zero evita NaN quando todos os
// valores são iguais, mas nesse caso o atributo não contém variação útil. Em
// produção, limites devem vir do conjunto de treino, ser persistidos e ser
// aplicados sem recalcular ao atender usuários novos.
const normalize = (value, min, max) => (value - min) / ((max - min) || 1)

function makeContext(products, users) {
    // O contexto guarda os dados e as tabelas necessárias para transformar
    // produtos/usuários em vetores. Vocabulários de cor e categoria são criados
    // a partir do catálogo atual; a posição depende da ordem dos dados. Para
    // servir o mesmo modelo depois, persista também estes vocabulários e os
    // limites de normalização. Catálogo vazio, usuários vazios, idades ausentes
    // ou categorias novas exigem validação e política explícita.
    const ages = users.map(u => u.age)
    const prices = products.map(p => p.price)

    const minAge = Math.min(...ages)
    const maxAge = Math.max(...ages)

    const minPrice = Math.min(...prices)
    const maxPrice = Math.max(...prices)

    const colors = [...new Set(products.map(p => p.color))]
    const categories = [...new Set(products.map(p => p.category))]

    const colorsIndex = Object.fromEntries(
        colors.map((color, index) => {
            return [color, index]
        }))
    const categoriesIndex = Object.fromEntries(
        categories.map((category, index) => {
            return [category, index]
        }))

    // MÉDIA DE IDADE DOS COMPRADORES: para cada produto, calcula a média das
    // idades dos usuários que o compraram e a usa como uma feature do produto.
    // Ex.: compras de pessoas com 20 e 40 anos produzem média 30. Sem compras,
    // usa o ponto médio das idades disponíveis. É uma heurística instável com
    // poucas compras; compras repetidas podem contar várias vezes e a associação
    // não prova que idade cause preferência. Dados de compra também refletem
    // preço, disponibilidade, exposição e outras variáveis não representadas.
    // Também há vazamento potencial: esta média é calculada com todas as compras
    // recebidas, inclusive as que viram rótulos positivos mais adiante. Se houver
    // avaliação temporal, calcule-a somente com compras anteriores ao corte.
    const midAge = (minAge + maxAge) / 2
    const ageSums = {}
    const ageCounts = {}

    users.forEach(user => {
        user.purchases.forEach(p => {
            ageSums[p.name] = (ageSums[p.name] || 0) + user.age
            ageCounts[p.name] = (ageCounts[p.name] || 0) + 1
        })
    })

    const productAvgAgeNorm = Object.fromEntries(
        products.map(product => {
            const avg = ageCounts[product.name] ?
                ageSums[product.name] / ageCounts[product.name] :
                midAge

            return [product.name, normalize(avg, minAge, maxAge)]
        })
    )

    return {
        products,
        users,
        colorsIndex,
        categoriesIndex,
        productAvgAgeNorm,
        minAge,
        maxAge,
        minPrice,
        maxPrice,
        numCategories: categories.length,
        numColors: colors.length,
        // price + age + colors + categories
        dimentions: 2 + categories.length + colors.length
    }
}

// One-hot representa uma categoria por um vetor: com 3 categorias, a segunda
// vira [0,1,0]. A multiplicação por `weight` reduz/aumenta a escala do grupo;
// ela não codifica similaridade entre categorias (preto não é numericamente
// mais próximo de cinza que de azul). Categorias desconhecidas precisam ser
// mapeadas para "outros" ou tratadas explicitamente antes de chamar tf.oneHot.
const oneHotWeighted = (index, length, weight) =>
    tf.oneHot(index, length).cast('float32').mul(weight)

function encodeProduct(product, context) {
    // TRANSFORMAÇÃO DO PRODUTO: produz um vetor com [preço, idade média dos
    // compradores, categorias one-hot, cores one-hot], cada grupo escalado por
    // WEIGHTS. É uma representação baseada em conteúdo (metadados do item), não
    // um embedding aprendido a partir de texto ou comportamento. Itens com mesmo
    // conteúdo ficam indistinguíveis para este vetor, mesmo que seus nomes ou
    // funções sejam diferentes. Os pesos e a seleção de atributos precisam de
    // avaliação; não são verdades descobertas pela rede.
    const price = tf.tensor1d([
        normalize(
            product.price,
            context.minPrice,
            context.maxPrice
        ) * WEIGHTS.price
    ])

    const age = tf.tensor1d([
        (
            context.productAvgAgeNorm[product.name] ?? 0.5
        ) * WEIGHTS.age
    ])

    const category = oneHotWeighted(
        context.categoriesIndex[product.category],
        context.numCategories,
        WEIGHTS.category
    )

    const color = oneHotWeighted(
        context.colorsIndex[product.color],
        context.numColors,
        WEIGHTS.color
    )

    return tf.concat1d(
        [price, age, category, color]
    )
}

function encodeUser(user, context) {
    if (user.purchases.length) {
        // PERFIL COM HISTÓRICO: calcula a média dos vetores dos produtos comprados.
        // Ex.: uma compra da categoria A e outra da B gera 0.5 em cada posição
        // categórica (antes do peso). A média perde ordem, quantidade e tempo:
        // duas compras antigas pesam tanto quanto duas recentes. Mais importante,
        // este ramo não codifica diretamente `user.age`; usa a média das features
        // dos produtos comprados, incluindo a idade média associada a esses itens.
        // É uma representação simples de preferência, inadequada se recência,
        // frequência ou sequência forem essenciais.
        return tf.stack(
            user.purchases.map(
                product => encodeProduct(product, context)
            )
        )
            .mean(0)
            .reshape([
                1,
                context.dimentions
            ])
    }

    // COLD START (usuário sem compras): não há histórico para inferir gosto, então
    // o vetor contém idade e zeros em preço/categoria/cor. Cold start é a falta
    // de informação sobre um usuário novo (ou, em outros sistemas, um item novo).
    // Este ramo trata somente usuários sem compras. O modelo ainda pode pontuar
    // itens, mas sem histórico a personalização é limitada e tende a depender de
    // idade ou dos vieses aprendidos no catálogo.
    return tf.concat1d(
        [
            tf.zeros([1]), // preço é ignorado,
            tf.tensor1d([
                normalize(user.age, context.minAge, context.maxAge)
                * WEIGHTS.age
            ]),
            tf.zeros([context.numCategories]), // categoria ignorada,
            tf.zeros([context.numColors]), // color ignorada,

        ]
    ).reshape([1, context.dimentions])
}

function createTrainingData(context) {
    // Cria exemplos binários (usuário, produto): 1 se o produto aparece nas
    // compras, 0 caso contrário. Isso é feedback implícito: uma compra é um
    // sinal de interesse, mas ausência de compra NÃO prova desinteresse; talvez
    // o usuário nunca tenha visto o item. Aqui todos os itens não comprados são
    // tratados como negativos presumidos, sem amostragem ou informação de
    // exposição; alguns desses rótulos 0 podem, portanto, estar errados.
    // Só usuários com ao menos uma compra entram no treino, então usuários novos
    // não fornecem exemplos. O tamanho cresce como usuários × produtos.
    //
    // LIMITAÇÃO CRÍTICA: `userVector` é calculado com todas as compras do usuário
    // e depois comparado com cada produto, inclusive os próprios comprados que
    // viram rótulos positivos. Assim, a feature do usuário contém o item-alvo:
    // há vazamento de alvo (target leakage) e os exemplos positivos ficam
    // artificialmente fáceis. Uma avaliação correta deve separar eventos por
    // tempo: construir o perfil só com histórico anterior ao evento que se quer
    // prever e testar em compras posteriores. Esta demo não faz essa separação.
    const inputs = []
    const labels = []
    context.users
        .filter(u => u.purchases.length)
        .forEach(user => {
            const userVector = encodeUser(user, context).dataSync()
            context.products.forEach(product => {
                const productVector = encodeProduct(product, context).dataSync()

                const label = user.purchases.some(
                    purchase => purchase.name === product.name ?
                        1 :
                        0
                )
                // A rede recebe a concatenação do vetor de usuário e do produto;
                // assim aprende um score para cada par, em vez de apenas uma
                // preferência global por produto.
                inputs.push([...userVector, ...productVector])
                labels.push(label)

            })
        })

    return {
        xs: tf.tensor2d(inputs),
        ys: tf.tensor2d(labels, [labels.length, 1]),
        inputDimention: context.dimentions * 2
        // tamanho = userVector + productVector
    }
}

// ====================================================================
// 📌 Exemplo de como um usuário é ANTES da codificação
// ====================================================================
/*
const exampleUser = {
    id: 201,
    name: 'Rafael Souza',
    age: 27,
    purchases: [
        { id: 8, name: 'Boné Estiloso', category: 'acessórios', price: 39.99, color: 'preto' },
        { id: 9, name: 'Mochila Executiva', category: 'acessórios', price: 159.99, color: 'cinza' }
    ]
};
*/

// EXEMPLO DE CODIFICAÇÃO: suponha categorias [acessórios, eletrônicos, vestuário]
// e cores [preto, cinza, azul]. O one-hot de acessórios é [1,0,0]; o de preto
// é [1,0,0]. Se Rafael comprou um item preto e outro cinza, `encodeUser` tira a
// média: a parte de cor fica [0.5,0.5,0], depois multiplicada pelo peso de cor.
// Essa média é um resumo, não uma compreensão semântica: não sabe que preto e
// cinza são cores parecidas, nem que uma compra foi presente. O modelo recebe
// números e os metadados que escolhemos codificar; nomes e descrições textuais
// não participam do aprendizado nesta implementação.



// ====================================================================
// 🧠 Configuração e treinamento da rede neural
// ====================================================================
async function configureNeuralNetAndTrain(trainData) {

    const model = tf.sequential()
    // REDE NEURAL DENSA para classificação binária do par (usuário, produto).
    // `inputShape` é o número de features concatenadas. `units` conta as saídas
    // aprendidas da camada, não são detectores independentes de conceitos.
    // ReLU = max(0,x) introduz não linearidade. As camadas 128→64→32 são uma
    // arquitetura escolhida para a demonstração, sem busca de hiperparâmetros.
    // Muitos pesos em relação aos exemplos elevam o risco de sobreajuste; um
    // baseline de popularidade, regras simples ou similaridade de conteúdo pode
    // ser mais apropriado quando há pouco histórico. Redes ficam mais justificadas
    // quando há dados suficientes, sinal útil e ganho comprovado em teste.
    model.add(
        tf.layers.dense({
            inputShape: [trainData.inputDimention],
            units: 128,
            activation: 'relu'
        })
    )
    // Cada camada densa calcula novas combinações das features anteriores. Ter
    // menos unidades (64 e depois 32) não garante uma compressão útil nem que
    // sejam retidos os padrões "mais importantes"; isso só seria sustentado por
    // resultados de validação e análise, ausentes nesta demo.
    model.add(
        tf.layers.dense({
            units: 64,
            activation: 'relu'
        })
    )

    // Segunda camada oculta. Sua largura é outro hiperparâmetro: alterar 32,
    // 64 ou 128 muda capacidade e custo, mas não há valor universalmente correto.
    model.add(
        tf.layers.dense({
            units: 32,
            activation: 'relu'
        })
    )
    // Camada de saída
    // SIGMOIDE transforma a saída em score entre 0 e 1. Com rótulos 0/1 e
    // entropia cruzada binária, pode ser interpretado como estimativa de classe
    // positiva, mas não é automaticamente uma probabilidade calibrada. Aqui,
    // devido aos negativos presumidos, ao vazamento e à falta de validação, não
    // se deve interpretar 0.8 como "80% de chance de comprar".
    model.add(
        tf.layers.dense({ units: 1, activation: 'sigmoid' })
    )

    // Adam é um otimizador adaptativo; `0.01` é a taxa de aprendizado, que
    // controla o tamanho das atualizações. Valor alto pode oscilar ou divergir e
    // valor baixo pode aprender devagar. Deve ser ajustado com validação, não
    // escolhido pela aparência do gráfico de treino.
    model.compile({
        optimizer: tf.train.adam(0.01),
        loss: 'binaryCrossentropy',
        // Accuracy é proporção de rótulos classificados corretamente no limiar
        // padrão. Se houver muitos zeros e poucos 1s, prever sempre 0 pode dar
        // accuracy alta e recomendações inúteis; acompanhe precisão/recall,
        // ranking (por exemplo, Recall/NDCG@K) e compare com baseline.
        metrics: ['accuracy']
    })

    // Treina por 100 épocas, lotes de 32. Sem conjunto de validação, não há
    // early stopping nem evidência de generalização. A acurácia enviada pela
    // callback é de treino; ela pode subir enquanto o desempenho real piora.
    await model.fit(trainData.xs, trainData.ys, {
        epochs: 100,
        batchSize: 32,
        shuffle: true,
        callbacks: {
            onEpochEnd: (epoch, logs) => {
                postMessage({
                    type: workerEvents.trainingLog,
                    epoch: epoch,
                    loss: logs.loss,
                    accuracy: logs.acc
                });
            }
        }
    })

    return model
}
async function trainModel({ users }) {
    console.log('Training model with users:', users);
    postMessage({ type: workerEvents.progressUpdate, progress: { progress: 1 } });
    const products = await (await fetch('/data/products.json')).json()

    const context = makeContext(products, users)
    context.productVectors = products.map(product => {
        return {
            name: product.name,
            meta: { ...product },
            vector: encodeProduct(product, context).dataSync()
        }
    })

    _globalCtx = context

    const trainData = createTrainingData(context)
    _model = await configureNeuralNetAndTrain(trainData)

    postMessage({ type: workerEvents.progressUpdate, progress: { progress: 100 } });
    postMessage({ type: workerEvents.trainingComplete });
}
function recommend({ user }) {
    if (!_model) return;
    const context = _globalCtx
    // 1. Codifica o usuário no mesmo espaço numérico usado no treino. Com
    // histórico, o vetor é a média de seus produtos comprados; sem histórico,
    // só idade é representada. Portanto, os comentários sobre atributos do
    // usuário não significam que nome ou perfil demográfico completo sejam usados.

    const userVector = encodeUser(user, context).dataSync()

    // Em catálogos grandes, pontuar cada item pode ser caro. Um sistema costuma
    // recuperar candidatos (candidate generation) e depois reordená-los (ranking).
    // Busca vetorial só faz sentido se os vetores e a métrica de distância forem
    // treinados/validados para recuperar itens relevantes; proximidade dos vetores
    // manuais daqui não garante interesse. Para este catálogo pequeno, pontuar
    // todos os produtos é simples e evita adicionar uma infraestrutura desnecessária.

    // 2. Forma um par [vetor do usuário, vetor do produto] para cada item. O
    // modelo retorna score de compatibilidade aprendido a partir dos rótulos
    // desta demo; ele não é uma verdade objetiva sobre gosto ou compra.


    const inputs = context.productVectors.map(({ vector }) => {
        return [...userVector, ...vector]
    })

    // 3. Tensor de formato [quantidade de produtos, features do par]. Processar
    // vários itens em lote evita uma chamada predict por produto e é adequado
    // para catálogo pequeno; lotes menores podem ser necessários em grande escala.
    const inputTensor = tf.tensor2d(inputs)

    // 4. Calcula um score por produto e ordena por ele. Score maior significa
    // apenas que a rede prefere esse par segundo os exemplos de treino. Não
    // equivale a probabilidade calibrada de compra: além do treino minúsculo e do
    // vazamento, os produtos não comprados foram tratados como negativos mesmo
    // sem sabermos se foram exibidos. Esse é o viés de exposição típico de
    // feedback implícito. Uma solução real precisa registrar impressões/cliques,
    // escolher negativos com cuidado e avaliar ranking em eventos futuros.
    const predictions = _model.predict(inputTensor)

    // 5. `dataSync` copia valores do tensor para um array JS de forma síncrona;
    // em WebGL isso pode bloquear enquanto aguarda a GPU. Para grande volume,
    // prefira a variante assíncrona e processe em lotes.
    const scores = predictions.dataSync()

    // Libera estes dois tensores temporários. O ciclo didático também cria
    // tensores auxiliares em encodeProduct/encodeUser; em produção, acompanhe e
    // descarte todos os tensores que não forem mais necessários para não crescer
    // o uso de memória WebGL/WASM ao treinar ou atender muitas requisições.
    inputTensor.dispose()
    predictions.dispose()

    const recommendations = context.productVectors.map((item, index) => {
        return {
            ...item.meta,
            name: item.name,
            score: scores[index] // previsão do modelo para este produto
        }
    })

    const sortedItems = recommendations
        .sort((a, b) => b.score - a.score)

    // 6. Envia a lista ordenada à thread principal para exibição. A ordenação
    // só reflete scores deste modelo e deste conjunto de treino; não garante
    // diversidade, novidade, disponibilidade nem satisfação do usuário.
    postMessage({
        type: workerEvents.recommend,
        user,
        recommendations: sortedItems
    });

}
const handlers = {
    [workerEvents.trainModel]: trainModel,
    [workerEvents.recommend]: recommend,
};

self.onmessage = e => {
    const { action, ...data } = e.data;
    if (handlers[action]) handlers[action](data);
};
