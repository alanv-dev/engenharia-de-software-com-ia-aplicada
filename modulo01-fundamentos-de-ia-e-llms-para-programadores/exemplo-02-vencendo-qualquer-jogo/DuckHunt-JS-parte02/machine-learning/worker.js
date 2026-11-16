importScripts('https://cdn.jsdelivr.net/npm/@tensorflow/tfjs@latest');

const MODEL_PATH = `yolov5n_web_model/model.json`;
const LABELS_PATH = `yolov5n_web_model/labels.json`;
const INPUT_MODEL_DIMENTIONS = 640
const CLASS_THRESHOLD = 0.4

let _labels = []
let _model = null
async function loadModelAndLabels() {
    await tf.ready()

    _labels = await (await fetch(LABELS_PATH)).json()
    _model = await tf.loadGraphModel(MODEL_PATH)

    // A inferência de aquecimento (warm-up) executa o modelo uma vez com uma
    // entrada fictícia. Algumas operações do TensorFlow.js inicializam kernels
    // ou escolhem um backend na primeira execução; fazê-lo aqui evita que essa
    // inicialização pese sobre o primeiro frame do jogo. A entrada precisa ter
    // o shape esperado pelo modelo. Esta execução não treina nem melhora o modelo.
    // ATENÇÃO: `executeAsync` também retorna tensores de saída. Este exemplo não
    // os descarta no aquecimento; num ciclo de vida rigoroso, guarde o retorno e
    // chame `dispose()` em cada tensor, como `runInference` faz abaixo.
    const dummyInput = tf.ones(_model.inputs[0].shape)
    await _model.executeAsync(dummyInput)
    tf.dispose(dummyInput)

    postMessage({ type: 'model-loaded' })

}

/**
 * PRÉ-PROCESSAMENTO — converte a imagem recebida no tensor que o modelo espera.
 * O contrato exato (tamanho, canais, escala e ordem RGB/BGR) vem do processo
 * usado para exportar o modelo; trocar o arquivo do modelo pode exigir mudar
 * estas operações. Uma entrada com formato incompatível pode falhar ou produzir
 * detecções sem sentido, mesmo que a inferência termine sem erro.
 *
 * 1. `tf.browser.fromPixels` transforma os pixels em tensor [altura, largura,
 *    canais]. Este código espera três canais de cor.
 * 2. `resizeBilinear` interpola pixels para 640 × 640. Interpolação bilinear
 *    calcula cada pixel novo a partir de vizinhos e suaviza a imagem. Como o
 *    código força um quadrado, uma imagem retangular fica deformada. Muitos
 *    modelos YOLO são treinados com letterboxing: redimensionar preservando a
 *    proporção e preencher margens. Se este modelo espera letterboxing, a
 *    transformação atual altera a imagem e também exige desfazer as margens ao
 *    converter as caixas de volta.
 * 3. `.div(255)` supõe pixels inteiros de 0 a 255 e os escala para 0 a 1.
 *    Outros modelos podem esperar [-1, 1], outra média/desvio ou outra ordem
 *    de canais. Não escolha a normalização por conveniência: reproduza o treino.
 * 4. `expandDims(0)` adiciona a dimensão de lote (batch), produzindo [1, 640,
 *    640, 3]. O primeiro `1` significa que processamos um frame por chamada.
 *
 * `tf.tidy` descarta tensores temporários criados dentro do callback. O tensor
 * retornado é mantido; `runInference` assume a responsabilidade por descartá-lo.
 */
function preprocessImage(input) {
    return tf.tidy(() => {
        const image = tf.browser.fromPixels(input)

        return tf.image
            .resizeBilinear(image, [INPUT_MODEL_DIMENTIONS, INPUT_MODEL_DIMENTIONS])
            .div(255)
            .expandDims(0)
    })
}

async function runInference(tensor) {
    const output = await _model.executeAsync(tensor)
    tf.dispose(tensor)
    // CONTRATO DE SAÍDA: o `model.json` incluído declara quatro saídas e contém
    // operações de NMS (supressão de caixas sobrepostas), portanto essa etapa
    // já ocorre dentro do grafo deste modelo. Este código interpreta as três
    // primeiras como (1) caixas, (2) scores e (3) IDs
    // de classe; ignora a quarta, embora descarte seus tensores abaixo. Essa
    // correspondência é específica da exportação incluída, não uma convenção
    // universal do YOLO. Antes de trocar o modelo, confira `_model.outputs`,
    // nomes, shapes e valores de uma inferência conhecida. Outros exports podem
    // devolver uma matriz única, caixas [centroX, centroY, largura, altura],
    // coordenadas em pixels ou exigir pós-processamento/NMS adicional. Interpretar uma
    // saída com o formato errado desloca caixas ou associa scores/classes aos
    // objetos errados.

    const [boxes, scores, classes] = output.slice(0, 3)
    const [boxesData, scoresData, classesData] = await Promise.all(
        [
            boxes.data(),
            scores.data(),
            classes.data(),
        ]
    )

    output.forEach(t => t.dispose())

    return {
        boxes: boxesData,
        scores: scoresData,
        classes: classesData
    }
}

/**
 * PÓS-PROCESSAMENTO — converte saídas numéricas em alvos que o jogo consegue
 * usar. Um bounding box (caixa delimitadora) descreve a região de um objeto.
 * Este código assume que cada caixa tem quatro valores [x1, y1, x2, y2], isto
 * é, cantos superior esquerdo e inferior direito normalizados para [0, 1], e
 * que `scores[i]` e `classes[i]` descrevem essa mesma caixa. Verifique esse
 * contrato no export do modelo: saídas em pixels ou no formato centro/largura/
 * altura exigem conversão diferente.
 *
 * O limiar `CLASS_THRESHOLD` elimina pontuações abaixo de 0.4. Aumentá-lo tende
 * a reduzir falsos positivos (alvos inexistentes), mas também pode perder
 * objetos reais; diminuí-lo tende a fazer o contrário. Escolha-o usando exemplos
 * rotulados e uma métrica que reflita o custo de cada erro. O score do detector
 * é uma medida interna de confiança, não necessariamente uma probabilidade
 * calibrada de que a classe esteja correta.
 *
 * O generator (`function*`) produz uma caixa por vez quando o chamador itera
 * sobre ele. Isso evita construir outra lista completa; o chamador, porém,
 * ainda envia cada predição por `postMessage`.
 */
function* processPrediction({ boxes, scores, classes }, width, height) {
    for (let index = 0; index < scores.length; index++) {
        if (scores[index] < CLASS_THRESHOLD) continue

        // DECISÃO DIDÁTICA: o detector genérico foi treinado para classes do COCO,
        // não para estes sprites de Duck Hunt. Uma silhueta triangular pode ser
        // classificada como `kite` (pipa), e o exemplo usa essa classe como
        // substituta para acionar o jogo. Isso demonstra inferência e integração,
        // não reconhecimento confiável de patos: objetos parecidos também podem
        // passar pelo filtro e patos desenhados de outra forma podem ser perdidos.
        // Para resolver o problema de verdade, reúna imagens rotuladas dos sprites
        // nas condições do jogo, avalie um detector apropriado/fine-tuned e meça
        // falsos positivos e falsos negativos em cenas não usadas no treino.
        const label = _labels[classes[index]]
        if (label !== 'kite') continue

        // Este cálculo só é correto para [x1,y1,x2,y2] normalizados em relação
        // ao frame original. Multiplicar pelas dimensões leva coordenadas 0–1 a
        // pixels; se o modelo retornar pixels 0–640, não se deve multiplicar
        // dessa forma. Com letterboxing, também seria preciso remover o padding.
        let [x1, y1, x2, y2] = boxes.slice(index * 4, (index + 1) * 4)
        x1 *= width
        x2 *= width
        y1 *= height
        y2 *= height

        const boxWidth = x2 - x1
        const boxHeight = y2 - y1
        const centerX = x1 + boxWidth / 2
        const centerY = y1 + boxHeight / 2

        // O centro da caixa é usado como ponto de mira. O score é formatado como
        // texto percentual apenas para exibição; arredondamento e multiplicação
        // por 100 não o tornam uma probabilidade estatisticamente calibrada.
        yield {
            x: centerX,
            y: centerY,
            score: (scores[index] * 100).toFixed(2)
        }

    }
}

loadModelAndLabels()

self.onmessage = async ({ data }) => {
    if (data.type !== 'predict') return
    if (!_model) return

    // A imagem chega como ImageBitmap transferido pela thread principal. Após
    // transferir, o worker é responsável por liberar esse recurso nativo. Este
    // exemplo não chama `close()`; numa aplicação que processa muitos frames,
    // feche o bitmap depois que `fromPixels` terminar de ler seus pixels para
    // evitar retenção de memória gráfica.
    const input = preprocessImage(data.image)
    const { width, height } = data.image

    const inferenceResults = await runInference(input)

    for (const prediction of processPrediction(inferenceResults, width, height)) {
        postMessage({
            type: 'prediction',
            ...prediction
        });
    }
};

console.log('🧠 YOLOv5n Web Worker initialized');
