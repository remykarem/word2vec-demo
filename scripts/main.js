window.tokens = []
window.negSampling = false

var T, opt;

var Y; // tsne result stored here
var data;

function updateEmbedding() {

    // get current solution
    var Y = T.getSolution();
    // move the groups accordingly
    gs.attr("transform", function (d, i) {
        return "translate(" +
            ((Y[i][0] * 20 * ss + tx) + 300) + "," +
            ((Y[i][1] * 20 * ss + ty) + 200) + ")";
    });
}

var svg;
function initEmbedding() {
    $("#embed").empty();
    var div = d3.select("#embed");
    svg = div.append("svg") // svg is global
        .attr("width", 600)
        .attr("height", 400);
}

var gs;
var cs;
var ts;
function drawEmbedding() {

    gs = svg.selectAll(".b")
        .data(data)
        .enter().append("g")
        .attr("class", "u");

    cs = gs.append("circle")
        .attr("cx", 0)
        .attr("cy", 0)
        .attr("r", 5)
        .attr('stroke-width', 1)
        .attr('stroke', 'black')
        .attr('fill', 'rgb(100,100,255)');

    if (labels.length > 0) {
        ts = gs.append("text")
            .attr("text-anchor", "top")
            .attr("transform", "translate(5, -5)")
            .attr("font-size", 12)
            .attr("fill", "#333")
            .text(function (d, i) { return labels[i]; });
    }

    var zoomListener = d3.behavior.zoom()
        .scaleExtent([0.1, 10])
        .center([0, 0])
        .on("zoom", zoomHandler);
    zoomListener(svg);
}

var tx = 0, ty = 0;
var ss = 1;
function zoomHandler() {
    tx = d3.event.translate[0];
    ty = d3.event.translate[1];
    ss = d3.event.scale;
}

var stepnum = 0;
function step() {
    if (dotrain) {
        var cost = T.step(); // do a few steps
        $("#cost").html("iteration " + T.iter + ", cost: " + cost);
    }
    updateEmbedding();
}

labels = [];
function preProLabels() {
    var lines = tokens;
    labels = [];
    for (var i = 0; i < lines.length; i++) {
        var row = lines[i];
        if (! /\S/.test(row)) {
            // row is empty and only has whitespace
            continue;
        }
        labels.push(row);
    }
}

dataok = false;
function preProData() {
    dataok = true;
    data = showAllEmbeddings();
    // var txt = $("#incsv").val();
    // var d = $("#deltxt").val();
    // var lines = txt.split("\n");
    // var raw_data = [];
    // var dlen = -1;
    // for (var i = 0; i < lines.length; i++) {
    //     var row = lines[i];
    //     if (! /\S/.test(row)) {
    //         // row is empty and only has whitespace
    //         continue;
    //     }
    //     var cells = row.split(d);
    //     var data_point = [];
    //     for (var j = 0; j < cells.length; j++) {
    //         if (cells[j].length !== 0) {
    //             data_point.push(parseFloat(cells[j]));
    //         }
    //     }
    //     var dl = data_point.length;
    //     if (i === 0) { dlen = dl; }
    //     if (dlen !== dl) {
    //         // TROUBLE. Not all same length.
    //         console.log('TROUBLE: row ' + i + ' has bad length ' + dlen);
    //         dlen = dl; // hmmm... 
    //         dataok = false;
    //     }
    //     raw_data.push(data_point);
    // }
    // data = raw_data; // set global
}

var dataset;

function createModel(vocabSize, embeddingSize = 10) {
    const input = tf.input({ shape: [vocabSize] });
    const dense1 = tf.layers.dense({ units: embeddingSize, useBias: false }).apply(input);
    const embedding = tf.model({ inputs: input, outputs: dense1 });
    const dense2 = tf.layers.dense({ units: vocabSize, useBias: false, activation: 'sigmoid' }).apply(embedding.outputs);
    var model = tf.model({ inputs: embedding.inputs, outputs: dense2 });
    return [embedding, model]
}

function createModelLogistic(vocabSize, embeddingSize = 10) {
    const input1 = tf.input({ shape: [vocabSize] });
    const input2 = tf.input({ shape: [vocabSize] });
    const input = tf.layers.add().apply([input1, input2]);
    const dense1 = tf.layers.dense({ units: embeddingSize, useBias: false }).apply(input);
    const embedding = tf.model({ inputs: [input1, input2], outputs: dense1 });
    const dense2 = tf.layers.dense({ units: 1, useBias: false, activation: 'softmax' }).apply(embedding.outputs);
    var model = tf.model({ inputs: embedding.inputs, outputs: dense2 });
    return [embedding, model]
}

function getDataPairs(model, str, c, binaryClassification, numNegSamples = 3) {
    var arr = str.toLowerCase().trim().replace(/\n/g, " ").replace(/[\.,()]/g, "").replace(/  /g, " ").replace(/\?/g, "").replace(/\'/g, "_").split(" ")
    var inputs = []
    var outputs = []
    var bins = []
    tokens = [...new Set(arr)]
    parse = (t) => tokens.map((w, i) => t.reduce((a, b) => b === w ? ++a : a, 0))
    var freq = parse(arr)
    var _freq = freq.map((el) => { return Math.pow(el, 3 / 4) });
    var _sum = _freq.reduce((a, b) => a + b);
    var wordDistribution = _freq.map((el) => { return el / _sum })
    const cumulativeSum = (sum => value => sum += value)(0);
    const cdf = wordDistribution.map(cumulativeSum);

    // Training data table header
    $('#trainingTable thead tr').remove();
    $('#tokenTable thead tr').remove();
    if (binaryClassification) {
        $("#trainingTable thead").append(`<tr>
        <th scope="col">No.</th>
        <th scope="col">Input 1</th>
        <th scope="col">Input 2</th>
        <th scope="col">Output</th>
      </tr>`);
        $("#tokenTable thead").append(`<tr>
            <th scope="col">No.</th>
            <th scope="col">Token</th>
            <th scope="col">Freq</th>
            <th scope="col">Prob (Noise)</th>
          </tr>`);
    } else {
        $("#trainingTable thead").append(`<tr>
        <th scope="col">No.</th>
        <th scope="col">Input</th>
        <th scope="col">Output</th>
      </tr>`);
        $("#tokenTable thead").append(`<tr>
      <th scope="col">No.</th>
      <th scope="col">Token</th>
      <th scope="col">Freq</th>
    </tr>`);
    }

    // Tokens table
    for (var i = 0; i < tokens.length; i++) {
        var tableRow =
            `<tr>
                  <th scope="row">` + (i + 1) + `</th>
                  <td>` + tokens[i] + `</td>
                  <td>` + freq[i] + `</td>`
        tableRow += binaryClassification ? `<td>` + Math.round(wordDistribution[i] * 1000) / 1000 + `</td>` : ""
        tableRow += `</tr>`
        $("#tokenTable tbody").append(tableRow)
    }

    var inputArr = []
    var input = "";
    var output = "";
    var counter = 0;
    for (var i = 0; i < arr.length; i++) {
        input = "";
        inputArr = [];
        for (var j = -c; j <= c; j++) {
            ind = i + j
            if (j != 0 && ind >= 0 && ind < arr.length) {

                if (model === 2) {
                    // CBOW
                    input += arr[ind] + " "
                    inputArr.push(ind)
                    continue
                } else {
                    // Skip-gram
                    input = arr[i]
                    output = arr[ind]
                }

                counter++;
                var tableRow =
                    `<tr>
                <th>` + counter + `</td>
                <td>` + input + `</td>
                <td>` + output + `</td>`
                tableRow += binaryClassification ? `<td>` + `1` + `</td>` : ""
                tableRow += `</tr>`

                $("#trainingTable tbody").append(tableRow)
                inputs.push(tokens.indexOf(input))
                outputs.push(tokens.indexOf(output))
                if (binaryClassification) { bins.push(1) }

                var k = binaryClassification ? 0 : numNegSamples;
                for (k; k < numNegSamples; k++) {

                    // Choose another word
                    var idx = cdf.filter(el => Math.random() >= el).length
                    output = tokens[idx]

                    counter++;
                    var tableRow =
                        `<tr>
                    <th>` + counter + `</td>
                    <td>` + input + `</td>
                    <td>` + output + `</td>`
                    tableRow += binaryClassification ? `<td>` + `0` + `</td>` : ""
                    tableRow += `</tr>`

                    $("#trainingTable tbody").append(tableRow)
                    inputs.push(tokens.indexOf(input))
                    outputs.push(idx)
                    bins.push(0)
                }


            }

        }
        if (model === 2) {
            // Initialise
            if (i === 0) {
                var inputTensors = tf.oneHot(tf.tensor1d(inputArr, 'int32'), tokens.length).mean(0).expandDims()
            } else {
                var _t = tf.oneHot(tf.tensor1d(inputArr, 'int32'), tokens.length).mean(0).expandDims()
                var inputTensors = inputTensors.concat(_t, 0)
            }
            output = arr[i]

            counter++;
            var tableRow =
                `<tr>
                <th>` + counter + `</td>
                <td>` + input + `</td>
                <td>` + output + `</td>
            </tr>`

            $("#trainingTable tbody").append(tableRow)

            outputs.push(tokens.indexOf(output))
        }
    }

    // Prepare tensors
    if (model === 1) {
        var inputTensors = tf.oneHot(tf.tensor1d(inputs, 'int32'), tokens.length);
    }
    var outputTensors = tf.oneHot(tf.tensor1d(outputs, 'int32'), tokens.length);
    var binTensors = tf.tensor1d(bins, 'int32')

    console.log(inputTensors.shape)
    console.log(outputTensors.shape)
    console.log(binTensors.shape)

    return {
        "inputTensors": inputTensors, "outputTensors": outputTensors, "binTensors": binTensors,
        "tokens": tokens, "vocabularySize": tokens.length
    }
}

function showTokens(tokenArr) {
    var str = ""
    for (var i = 0; i < tokenArr.length; i++) {
        str += tokenArr[i] + "\n"
    }
    return str
}

function showAllEmbeddings() {
    negSampling = $("#negativeSampling").is(':checked')

    indices = [...Array(dataset["vocabularySize"])].map((_, i) => i)
    t = tf.oneHot(tf.tensor1d(indices, 'int32'), dataset["vocabularySize"]);
    if (negSampling) {
        t0 = tf.zerosLike(t);
        arrs = embedding.predict([t, t0]).arraySync();
        t0.dispose();
    } else {
        arrs = embedding.predict(t).arraySync()
    }
    t.dispose();

    return arrs
}

function onTrainBegin() {

}

function onBatchEnd(batch, logs) {
    //    console.log('Accuracy', logs.acc);
    console.log('Loss', logs.loss);
}

function onEpochEnd(currEpoch, logs) {
    var totalEpochs = Number($("#epochs").val())
    var pctgComplete = (currEpoch + 1) / totalEpochs * 100
    $("#progressBar").attr("style", "width: " + pctgComplete + "%")
}

function onTrainEnd() {
    $("#progressBar").removeClass("progress-bar-animated bg-warning")
}

function prepare() {

    // Get configs
    var model = Number($("#model").val());
    var windowSize = Number($("#windowSize").val());
    negSampling = $("#negativeSampling").is(':checked');
    var negatives = negSampling ? Math.floor(Number($("#negSamplingNegatives").val()) / Number($("#negSamplingPositives").val())) : -1;

    // Get data
    var textData = $("#textdata").val()

    // Clear tables
    $("#trainingTable tbody").empty()
    $("#tokenTable tbody").empty()

    // Show generated data
    dataset = getDataPairs(model, textData, windowSize, negSampling, negatives)
    $("#inlabels").val(showTokens(dataset["tokens"]))
    $("#statsVocabSize").text("Vocabulary size: " + dataset["tokens"].length)
    $("#statsNumExamples").text("No. of training examples: " + dataset["inputTensors"].shape[0])
    $("#perptxt").val(Math.floor(Math.sqrt(dataset["inputTensors"].shape[0])))

    // Show example data
    $("#randomise").removeAttr("disabled");
    getRandomData();

    // Now we can train model
    $("#trainModel").removeAttr("disabled");
}

function getOptimiser(num, lr) {
    switch (num) {
        case 1:
            return tf.train.sgd(lr)
        case 2:
            return tf.train.momentum(lr, 0.9)
        case 3:
            return tf.train.adam(lr)
        default:
            return tf.train.rmsprop(lr)
    }
}

function trainModel() {
    // Get train configs
    var model = Number($("#model").val())
    var embeddingSize = Number($("#embeddingSize").val())
    var totalEpochs = Number($("#epochs").val())
    var learningRate = Number($("#learningRate").val())
    var optimiserEnum = Number($("#inputGroupSelect01").val())
    var optimiser = getOptimiser(optimiserEnum, learningRate)
    var negSampling = $("#negativeSampling").is(':checked')

    $("#progressBar").addClass("progress-bar-animated bg-warning")
    $("#progressBar").attr("style", "width: 0%")

    if (model === 1 && negSampling) {

        // Skip-gram with negative sampling
        embedding_and_model = createModelLogistic(dataset["vocabularySize"], embeddingSize)
        embedding = embedding_and_model[0]
        model = embedding_and_model[1]
        model.compile({
            optimizer: optimiser,
            loss: 'binaryCrossentropy',
            metrics: ['accuracy']
        });
        model.fit([dataset["inputTensors"], dataset["outputTensors"]], dataset["binTensors"], {
            epochs: totalEpochs,
            batchSize: 32,
            callbacks: { onTrainBegin, onBatchEnd, onEpochEnd, onTrainEnd }
        }).then(info => {
            // console.log('Final accuracy', info.history.acc);
            // console.log('Loss', info.history.loss);
            // $("#incsv").val(showAllEmbeddings())
            $("#inbut").removeAttr("disabled");
            $("#stopbut").removeAttr("disabled");
        });

    } else if (model === 1 || model === 2) {

        // SKip-gram without negative sampling
        embedding_and_model = createModel(dataset["vocabularySize"], embeddingSize)
        embedding = embedding_and_model[0]
        model = embedding_and_model[1]
        model.compile({
            optimizer: optimiser,
            loss: 'categoricalCrossentropy',
            metrics: ['accuracy']
        });
        model.fit(dataset["inputTensors"], dataset["outputTensors"], {
            epochs: totalEpochs,
            batchSize: 64,
            callbacks: { onTrainBegin, onBatchEnd, onEpochEnd, onTrainEnd }
        }).then(info => {
            // console.log('Final accuracy', info.history.acc);
            // console.log('Loss', info.history.loss);
            // $("#incsv").val(showAllEmbeddings())
            $("#inbut").removeAttr("disabled");
            $("#stopbut").removeAttr("disabled");
        });
    }
}

const negativeSamplingHTML = `<div class="col-sm-5">
<div class="input-group mb-3">
  <div class="input-group-prepend">
    <span class="input-group-text" id="">Ratio</span>
  </div>
  <input type="text" id="negSamplingPositives" class="form-control" placeholder="Positives" value="1">
  <input type="text" id="negSamplingNegatives" class="form-control" placeholder="Negatives" value="3">
</div>
</div>`

function getRandomData() {
    var randIdx = Math.floor(Math.random() * dataset["vocabularySize"]);
    var sample = "Training example " + (randIdx + 1) + "\n\n";

    if (negSampling) {
        sample += "Input 1: ";
        sample += tokens[dataset["inputTensors"].arraySync()[randIdx].indexOf(1)] + "\n";
        sample += "[" + dataset["inputTensors"].arraySync()[randIdx].join(",") + "]\n";
        sample += "Input 2: ";
        sample += tokens[dataset["outputTensors"].arraySync()[randIdx].indexOf(1)] + "\n";
        sample += "[" + dataset["outputTensors"].arraySync()[randIdx].join(",") + "]\n";
        sample += "Output:\n";
        sample += "[" + dataset["binTensors"].arraySync()[randIdx] + "]\n";
    } else if (model === 1 && !negSampling) {
        sample += "Input: ";
        sample += tokens[dataset["inputTensors"].arraySync()[randIdx].indexOf(1)] + "\n";
        sample += "[" + dataset["inputTensors"].arraySync()[randIdx].join(",") + "]\n";
        sample += "Output: ";
        sample += tokens[dataset["outputTensors"].arraySync()[randIdx].indexOf(1)] + "\n";
        sample += "[" + dataset["outputTensors"].arraySync()[randIdx].join(",") + "]\n";
    } else {
        sample += "Input:\n";
        sample += "[" + dataset["inputTensors"].arraySync()[randIdx].map(el => { return Math.round(el * 100) / 100 }).join(",") + "]\n";
        sample += "Output: ";
        sample += tokens[dataset["outputTensors"].arraySync()[randIdx].indexOf(1)] + "\n";
        sample += "[" + dataset["outputTensors"].arraySync()[randIdx].join(",") + "]\n";
    }

    $("code").text(sample)
}

dotrain = true;
iid = -1;
$(window).load(() => {

    initEmbedding();

    $("#model").on('change', function () {
        if (this.value === '1') {
            $("#negativeSampling").removeAttr("disabled");
            $("#modelDetails").text("This model trains words to predict their context.")
        } else {
            $("#negativeSampling")
                .prop("checked", false)
                .attr("disabled", "disabled");
            $("#negativeSamplingRatio div").remove();
            $("#modelDetails").text("This model trains multiple words (in the form of bag-of-words) surrounding a target word.")
            $("#trainDetails").text("Model will be trained with categorical cross entropy loss function.")
        }
    });

    $("#negativeSampling").change(function () {
        if (this.checked) {
            $("#negativeSamplingRatio").append(negativeSamplingHTML);
            $("#trainDetails").text("Model will be trained with binary cross entropy loss function.")
        } else {
            $("#negativeSamplingRatio div").remove();
            $("#trainDetails").text("Model will be trained with categorical cross entropy loss function.")
        }
    });

    $("#prepareData").click(() => { prepare() });

    $("#randomise").click(() => { getRandomData() });

    $("#trainModel").click(() => { trainModel() });

    $("#stopbut").click(() => { dotrain = false; });

    $("#inbut").click(() => {

        initEmbedding();
        preProData();
        if (!dataok) { // this is so terrible... globals everywhere #fasthacking #sosorry
            alert('there was trouble with data, probably rows had different number of elements. See console for output.');
            return;
        }
        preProLabels();
        if (labels.length > 0) {
            if (data.length !== labels.length) {
                alert('number of rows in Text labels (' + labels.length + ') does not match number of rows in Data (' + data.length + ')! Aborting.');
                return;
            }
        }

        // ok lets do this
        opt = {
            epsilon: parseFloat($("#lrtxt").val()),
            perplexity: parseInt($("#perptxt").val()),
            dim: data[0].length
        };
        T = new tsnejs.tSNE(opt); // create a tSNE instance

        var dfv = 'raw';
        // var dfv = $('input[name=rdata]:checked', '#datatypeform').val();
        if (dfv === 'raw') {
            console.log('raw');
            T.initDataRaw(data);
        }
        if (dfv === 'dist') {
            console.log('dist');
            T.initDataDist(data);
        }
        drawEmbedding();
        iid = setInterval(step, 10);
        dotrain = true;

    });
});