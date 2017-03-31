const plotly = require('plotly')('leiverandres', 'qx7i39YxqtAXboq5uuQp');
const fs = require('fs');


function generateChart(cpu_data, gpu_data, labels, opts) {
  const image_name = opts.figure_name + '.png';
  let figure = {
    data: [{
      x: labels,
      y: cpu_data,
      type: 'scatter',
      name: opts.set1_name
    }, {
      x: labels,
      y: gpu_data,
      type: 'scatter',
      name: opts.set2_name
    }]
  };

  let imgOptions = {
    format: 'png', 
    width: 800, 
    height: 800,
    title: opts.figure_name,
    layout: {
      xaxis: {
        title: 'Tamaño de las matrices' 
      },
      yaxis: {
        title: 'Tiempo de ejecución (s)'
      }
    }
  };
  plotly.getImage(figure, imgOptions, function (err, imageStream) {
    if (err) return console.log(err);
    const fileStream = fs.createWriteStream(image_name);
    imageStream.pipe(fileStream);
    console.log(`Image generated: ${image_name}`);
  }); 
}

module.exports = generateChart;


