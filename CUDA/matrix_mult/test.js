const exec = require('child_process').execSync;
const clear = require('clear');
const mathjs = require('mathjs');
const generateChart = require('./generateChart');


const set1_filename = process.argv[2]; // set 1 exec name
const set2_filename = process.argv[3]; // set 2 exec name
const num_tests = process.argv[4] || 1;
const set1_name = process.argv[5];
const set2_name = process.argv[6];
const disp_name = set1_name + '_vs_' + set2_name;

clear();
console.log(`Executable set 1 filename: ${set1_filename}`);
console.log(`Executable set 2 filename: ${set2_filename}`);
console.log(`Number of tests: ${num_tests}`);
let a_rows = 100;
let a_cols = 100;
let b_cols = 100;
let data1 = [];
let data2 = [];
let labels = [];

for (let i = 0; i < num_tests; i++) {
  let data1_times = [];
  let data2_times = [];
  for (let j = 0; j < 20; j++) {
    let stdout1 = exec(`${set1_filename} ${a_rows} ${a_cols} ${b_cols}`);
    let stdout2 = exec(`${set2_filename} ${a_rows} ${a_cols} ${b_cols}`);
    data1_times[j] = Number(stdout1.toLocaleString());
    data2_times[j] = Number(stdout2.toLocaleString());  
  }
  data1[i] = mathjs.mean(data1_times);
  data2[i] = mathjs.mean(data2_times);
  labels[i] = `${a_rows}x${a_cols}`;
  let acceleration = data1[i] / data2[i];
  a_rows += 500;
  a_cols += 500;
  b_cols += 500;
 
  console.log(`\nTest ${i+1} for ${labels[i]} matrices:`);
  console.log(`${set1_name} time: ${data1[i]}`);
  console.log(`${set1_name} standar deviation: ${mathjs.std(data1_times)}`);
  console.log(`${set2_name} time: ${data2[i]}`);
  console.log(`${set2_name} standar deviation: ${mathjs.std(data2_times)}`);
  console.log(`Acceleration ${acceleration}X`); 
}

console.log(`${set1_name} data: ${data1}`);
console.log(`${set2_name} data: ${data2}`);
console.log(`Tested with following matrix sizes: ${labels}`)

let opts = {
  figure_name: disp_name,
  set1_name: set1_name,
  set2_name: set2_name
}
generateChart(data1, data2, labels, opts);
