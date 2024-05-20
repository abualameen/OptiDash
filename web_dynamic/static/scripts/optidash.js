function generateTable() {
    const rows = document.getElementById('rows').value;
    const cols = document.getElementById('cols').value;
    let table = '<table id=tableclear>';
    console.log(`rows: ${rows}`);
    console.log(`cols: ${cols}`)
    // Header row with criteria names

    table += '<tr>';
    table += '<th>Objective Function</th>';
    table += `<td ><input type="text" placeholder="Enter objective function 1" id="objectiveFunction1"></td>`;
    table += `<td><input type="text" placeholder="Enter objective function 2" id="objectiveFunction2"></td>`;
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

function submitFormData() {
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
    tableData1.push(obj1);
    tableData1.push(obj2);
    

    $.ajax({
        type: 'POST',
        url: '/home',
        contentType: 'application/json;charset=UTF-8',
        data: JSON.stringify({ tableData, tableData1}),
        success: function (data) {
            $('#results-container').html('The best choice of your alternatives is: ' + data.best_alternative + ' ranks 1st with a performance score of ' + data.best_perf_score);
        },

        error: function (error) {
            console.error('Error:', error);
        }
    });
}