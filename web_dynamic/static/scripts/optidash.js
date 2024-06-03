
const socket = io('http://localhost:5006');

console.log('Socket.IO script loaded, trying to connect...');

socket.on('connect', () => {
    console.log('Successfully connected to the server via Socket.IO');
});


  

socket.on('update', (data) => {
    // console.log('Received update from server:', data);
    
    if (data.nxt_gen_fit3) {
        plotData1(data.nxt_gen_fit1, data.nxt_gen_fit2, data.nxt_gen_fit3);
    } else {
        plotData(data.nxt_gen_fit1, data.nxt_gen_fit2);
    }
           
});


function generateTable() {
    const rows = document.getElementById('rows').value;
    const cols = document.getElementById('cols').value;
    let table = '<table id=tableclear class=table>';
    console.log(`rows: ${rows}`);
    console.log(`cols: ${cols}`)
    // Header row with criteria names

    table += '<tr>';
    table += `<td><input type="text" placeholder="Enter objective function 1" id="objectiveFunction1"></td>`;
    table += `<td><input type="text" placeholder="Enter objective function 2" id="objectiveFunction2"></td>`;
    table += `<td><input type="text" placeholder="Enter objective function 3" id="objectiveFunction3"></td>`;
    table += '</tr>';

       
    table += '<tr>';
    for (let k = 0; k < cols; k++) {
        table += `<th><select name="objective" required id="type_${k}">`;
        table += '<option value="" disabled selected hidden>Objectives</option>';
        table += '<option value="Minimization">Minimazation</option>';
        table += '<option value="Maximization">Maximization</option>';
        table += '</select></th>';
    }
    table += '</tr>';

    table += '<tr>';
    table += `<td><input type="number" placeholder="Enter Population size" id="pop"></td>`;
    table += `<td><input type="number" placeholder="Enter Iteration Number" id="ita"></td>`;
    table += `<td><input type="number" placeholder="Enter Cross-over rate" id="cor"></td>`;
    table += '</tr>';
    
    table += '<tr>';
    table += `<td><input type="number" placeholder="Enter Cross-over coeficient" id="coc"></td>`;
    table += `<td><input type="number" placeholder="Enter Mutation rate" id="mr"></td>`;
    table += `<td><input type="number" placeholder="Enter Mutation coeficient" id="mc"></td>`;
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
    const tableData2 = [];
    const tableData3 = []
    for (let i = 0; i < rows; i++) {
        const rowData = [];
        for (let j = 0; j < cols; j++) {
            const cellValue = document.getElementById(`value_${i}_${j}`).value;
            rowData.push(cellValue);
        }
        tableData.push(rowData);
        // console.log(tableData)
    }

    
    const obj1 = document.getElementById("objectiveFunction1").value;
    const obj2 = document.getElementById("objectiveFunction2").value;
    const obj3 = document.getElementById("objectiveFunction3").value;
    if (obj1) tableData1.push(obj1);
    if (obj2) tableData1.push(obj2);
    if (obj3) tableData1.push(obj3);

    // const objectives = document.getElementById("objectives").value
    // tableData2.push(objectives)

    for (let p = 0; p < cols; p++) {
        const cellValue = document.getElementById(`type_${p}`).value;
        if (cellValue) tableData2.push(cellValue);
        // console.log(tableData)
    }
    
    const pop_size = document.getElementById("pop").value;
    const Iteration = document.getElementById("ita").value;
    const Cross_rate = document.getElementById("cor").value;

    const Cross_coef  = document.getElementById("coc").value;
    const Mutation_rate = document.getElementById("mr").value;
    const Mutation_coef = document.getElementById("mc").value;
    
    tableData3.push(pop_size);
    tableData3.push(Iteration);
    tableData3.push(Cross_rate);
    tableData3.push(Cross_coef);
    tableData3.push(Mutation_rate);
    tableData3.push(Mutation_coef);

    

    $.ajax({
        type: 'POST',
        url: '/home',
        contentType: 'application/json;charset=UTF-8',
        data: JSON.stringify({ tableData, tableData1, tableData2, tableData3}),
        success: function (data) {
            try {
                // Check if data is defined and has the expected properties
                if (data) {
                            // Prepare data arrays
                    let obj1Values = data.opti_front_obj1; 
                    let obj2Values = data.opti_front_obj2; 
                    let obj3Values = data.opti_front_obj3 || []; 
                    let params = data.opti_para;
                    
                    function roundToSignificantDigits(num, digits) {
                        return Number(num).toPrecision(digits);
                    }
            
                    // Check if all arrays have the same length
                    if (obj1Values.length === obj2Values.length && (obj3Values.length === 0 || obj3Values.length === obj1Values.length) && params.length === obj1Values.length) {
                        let resultHtml = `
                            <table border="1">
                                <thead>
                                    <tr>
                                        <th>Optimal Parameters</th>
                                        <th>Optimal Fitness Value 1</th>
                                        <th>Optimal Fitness Value 2</th>
                                        ${obj3Values.length > 0 ? '<th>Optimal Fitness Value 3</th>' : ''}
                                    </tr>
                                </thead>
                                <tbody>
                        `;
            
                        for (let i = 0; i < obj1Values.length; i++) {
                            resultHtml += `
                                <tr>
                                    <td>[${params[i].map(num => roundToSignificantDigits(num, 4)).join(', ')}]</td>
                                    <td>${roundToSignificantDigits(obj1Values[i], 4)}</td>
                                    <td>${roundToSignificantDigits(obj2Values[i], 4)}</td>
                                    ${obj3Values.length > 0 ? `<td>${roundToSignificantDigits(obj3Values[i], 4)}</td>` : ''}
                                </tr>
                            `;
                        }
            
                        resultHtml += `
                                </tbody>
                            </table>
                        `;
            
                        $('#results-container').html(resultHtml);
                    } else {
                        $('#results-container').html('<p>Error: Mismatch in data lengths.</p>');
                    }
                    // Your existing logic using data
                } else {
                    // Handle the case where data is undefined or does not have the expected properties
                    throw new Error("Unexpected response format or empty data");
                }
            } catch (error) {
                console.error("Error processing data:", error);
                // Optionally, show a user-friendly message
                alert("An error occurred while processing your request. Please check your entries and try again.");
            }
        },
        error: function(jqXHR, textStatus, errorThrown) {
            console.error("AJAX request failed:", textStatus, errorThrown);
            // Optionally, show a user-friendly message
            alert("Failed to retrieve data. Please try again later.");
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
