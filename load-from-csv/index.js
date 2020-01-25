/*
//-------------------------------------------------------------
// finetunes the modified mobilenet model in 5 training batches
// takes model, images and targets as arguments
//-------------------------------------------------------------
async fineTuneModifiedModel(model,images,targets)
{
function onBatchEnd(batch, logs) 
{
    console.log('Accuracy', logs.acc);
    console.log('CrossEntropy', logs.ce);
    console.log('All', logs);
}
console.log('Finetuning the model...');

await model.fit(images, targets, 
{
    epochs: 5,
    batchSize: 24,
    validationSplit: 0.2,
    callbacks: {onBatchEnd}

}).then(info => {
    console.log
    console.log('Final accuracy', info.history.acc);
    console.log('Cross entropy', info.ce);
    console.log('All', info);
    console.log('All', info.history['acc'][0]);
    
    for ( let k = 0; k < 5; k++) 
{
    this.traningMetrics.push({acc: 0, ce: 0 , loss: 0});

    this.traningMetrics[k].acc=info.history['acc'][k];
    this.traningMetrics[k].ce=info.history['ce'][k];
    this.traningMetrics[k].loss=info.history['loss'][k]; 
}
    images.dispose();
    targets.dispose();
    model.dispose();
});;

}
*/

//-------------------------------------------------------------
// calls parseImages() to populate imageSrc and targets as a list 
// 
//-------------------------------------------------------------
async loadCSV()
{ 
    this.parseImages(120);

    if (this.isImagesListPerformed)
    {
      this.openSnackBar("Training images are listed !","Close");
    }
    if (!this.isImagesListPerformed)
    {
      this.openSnackBar("Please reset the dataset to upload new CSV file !","Reset");
    }
}

// reset(){};

//-------------------------------------------------------------
// stores Image Src and Class info in CSV file
// populates the MatTable rows and paginator
// populates the targets as [1,0] uninfected, [0,1] parasitized
//-------------------------------------------------------------
parseImages(batchSize)
{
if (this.isImagesListed) 
{
    this.isImagesListPerformed=false;
    return;
}

let allTextLines = this.csvContent.split(/\r|\n|\r/);

const csvSeparator = ',';
const csvSeparator_2 = '.';

for ( let i = 0; i < batchSize; i++) 
{
    // split content based on comma
    const cols: string[] = allTextLines[i].split(csvSeparator);
    
    this.tableRows.push({ImageSrc: '', LabelX1: 0 , LabelX2: 0, Class: ''});

    if (cols[0].split(csvSeparator_2)[1]=="png") 
    {  
    
    if (cols[1]=="Uninfected") 
    { 
        this.label_x1.push(Number('1'));
        this.label_x2.push(Number('0'));

        this.tableRows[i].ImageSrc="../assets/"+ cols[0];
        this.tableRows[i].LabelX1=1;
        this.tableRows[i].LabelX2=0;
        this.tableRows[i].Class="Uninfected";
    } 

    if (cols[1]=="Parasitized") 
    { 
        this.label_x1.push(Number('0'));
        this.label_x2.push(Number('1'));
    
        this.tableRows[i].ImageSrc="../assets/"+ cols[0];
        this.tableRows[i].LabelX1=0;
        this.tableRows[i].LabelX2=1;
        this.tableRows[i].Class="Parasitized";
    } 

    } 
}
this.table.renderRows();
this.dataSource.paginator = this.paginator;

this.isImagesListed=true;
this.isImagesListPerformed=true;
}
