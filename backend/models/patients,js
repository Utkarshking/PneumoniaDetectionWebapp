const mongoose=require('mongoose');
const validator=require("validator");
const patient=new mongoose.Schema({
    FirstName:{
        type:String,
        required:true,
        minlength:3
    },
    LastName:{
        type:String,
        required:true,
        minlength:3
    },
    DOB:{
        type:String,
        required:true,
    },
    Gender:{
        type:String,
        required:true
    },
    Phone:{
        type:Number,
        required:true,
        minlength:10,
        maxlength:10,
        unique:[true,"Phone Number is already present"]
    },
    ReportPhoto:{
       type:String,
       required:true
    },
    PredictionResult:{
        type:String,
        required:true
    }
})
const Patientmodel=new mongoose.model("Patientmodel",patient);
module.exports=Patientmodel;
