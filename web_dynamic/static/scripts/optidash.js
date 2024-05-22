
const socket = io('http://localhost:5006');

console.log('Socket.IO script loaded, trying to connect...');

socket.on('connect', () => {
    console.log('Successfully connected to the server via Socket.IO');
});

socket.on('update', (data) => {
    console.log('Received update from server:', data);
    if (data.nxt_gen_fit3) {
        plotData1(data.nxt_gen_fit1, data.nxt_gen_fit2, data.nxt_gen_fit3);
    } else {
        plotData(data.nxt_gen_fit1, data.nxt_gen_fit2);
    }
           
});

function generateTable() {
    const rows = document.getElementById('rows').value;
    const cols = document.getElementById('cols').value;
    let table = '<table id=tableclear>';
    console.log(`rows: ${rows}`);
    console.log(`cols: ${cols}`)
    // Header row with criteria names

    table += '<tr>';
    table += `<td><input type="text" placeholder="Enter objective function 1" id="objectiveFunction1"></td>`;
    table += `<td><input type="text" placeholder="Enter objective function 2" id="objectiveFunction2"></td>`;
    table += `<td><input type="text" placeholder="Enter objective function 3" id="objectiveFunction3"></td>`;
    table += '</tr>';
    for (let j = 0; j < cols; j++) {
        const content = ["Criteria Name", "Lower Bound", "Upper Bound"][j];
        table += `<th>${content}</th>`;
    }
    table += '</tr>';

    // table += '<tr>';
    // for (let k = 0; k < cols; k++) {
    //     table += `<th><select name="criteriaTypes" required id="type_${k}">`;
    //     table += '<option value="" disabled selected hidden>Criteria Type</option>';
    //     table += '<option value="NonBeneficial">Non-Beneficial</option>';
    //     table += '<option value="Beneficial">Beneficial</option>';
    //     table += '</select></th>';
    // }
    // table += '</tr>';



    // Rows with input fields
    for (let i = 0; i < rows; i++) {
        table += '<tr>';
        for (let j = 0; j < cols; j++) {
            if (j==0){
                table += `<td><input type="text" id="value_${i}_${j}" placeholder="variable name"></td>`;
            }else {
                table += `<td><input type="number" id="value_${i}_${j}" placeholder="Value"></td>`;
            }
        }
        table += '</tr>';
    }


    table += '</table>';
    const tableContainer = document.getElementById('table-container');
    tableContainer.innerHTML = table;
}

function submitFormData(event) {
    event.preventDefault()
    const rows = document.getElementById('rows').value;
    const cols = document.getElementById('cols').value;

    // Collect table data
    const tableData = [];
    const tableData1 = [];
    for (let i = 0; i < rows; i++) {
        const rowData = [];
        for (let j = 0; j < cols; j++) {
            const cellValue = document.getElementById(`value_${i}_${j}`).value;
            rowData.push(cellValue);
        }
        tableData.push(rowData);
        // console.log(tableData)
    }

    // for (let k = 0; k < cols; k++) {
    //     const criteriaName = document.getElementById(`name_${k}`).value;
    //     const criteriaType = document.getElementById(`type_${k}`).value;
    //     tableData1.push(criteriaName);
    //     tableData1.push(criteriaType);
    // }
    
    const obj1 = document.getElementById("objectiveFunction1").value;
    const obj2 = document.getElementById("objectiveFunction2").value;
    const obj3 = document.getElementById("objectiveFunction3").value;
    if (obj1) tableData1.push(obj1);
    if (obj2) tableData1.push(obj2);
    if (obj3) tableData1.push(obj3);

    

    $.ajax({
        type: 'POST',
        url: '/home',
        contentType: 'application/json;charset=UTF-8',
        data: JSON.stringify({ tableData, tableData1}),
        success: function (data) {
            $('#results-container').html('The best choice of your alternatives is: ' + data.best_alternative + ' ranks 1st with a performance score of ' + data.best_perf_score);
            // Call plotData with the data returned from the server
        },

        error: function (error) {
            console.error('Error:', error);
        }
    });
}

// socket.on('update', (data) => {
//     console.log(data);
//     plotData(data.fit1, data.fit2);
// });

function plotData(fit1, fit2) {
    console.log('Plotting data:', fit1, fit2);
    const trace = {
        x: fit1,
        y: fit2,
        mode: 'markers',
        type: 'scatter'
    };
    const layout = {
        title: 'Pareto Front',
        xaxis: { title: 'Objective 1' },
        yaxis: { title: 'Objective 2' }
    };
    Plotly.newPlot('plot', [trace], layout);
}

function plotData1(fit1, fit2, fit3) {
    console.log('Plotting data:', fit1, fit2, fit3);
    const trace = {
        x: fit1,
        y: fit2,
        z: fit3,
        mode: 'markers',
        type: 'scatter3d'
    };
    const layout = {
        title: '3D Pareto Front',
        scene: {
            xaxis: { title: 'Objective 1' },
            yaxis: { title: 'Objective 2' },
            zaxis: { title: 'Objective 3' }
        }
    };
    Plotly.newPlot('plot', [trace], layout);
}