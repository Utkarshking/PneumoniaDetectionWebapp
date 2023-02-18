import React, { useState } from "react";
import "./Navbar.css";
import SpinnerWidget from "./SpinnerWidget";
import "mdb-react-ui-kit/dist/css/mdb.min.css";
import {
  MDBBtn,
  MDBContainer,
  MDBRow,
  MDBCol,
  MDBCard,
  MDBCardBody,
  MDBInput,
  MDBRadio,
} from "mdb-react-ui-kit";
import axios from "axios";
function Test() {
  const PatientDetails = {
    FirstName: "",
    LastName: "",
    DOB: "",
    Gender: "Male",
    Phone: "",
  };
  const PatientPredictionDetails = {
    FirstName: "No Data Avaiable",
    LastName: "No Data Available",
    DOB: "No Data Available",
    Gender: "No Data Available",
    Phone: "No Data Available",
    ReportPhoto:
      "https://previews.123rf.com/images/pavelstasevich/pavelstasevich1902/pavelstasevich190200120/124934975-no-image-available-icon-vector-flat.jpg",
    PredictionResult: "No Data Available",
  };

  // const BASEURL="https://pneumonia-webapp-backend.herokuapp.com/predict"
  const BASEURL = "http://localhost:8000/upload";
  const [PatientRegistration, setUserRegistration] = useState(PatientDetails);
  const [FileUpload, SetFileuploadComplete] = useState();
  const [PatientPrediction, SetPatientPredictionDetails] = useState(
    PatientPredictionDetails
  );
  const [isLoading, setIsLoading] = useState(false);
  // handler for handling the patients details
  const handleinput = (e) => {
    const { name, value } = e.target;
    setUserRegistration({ ...PatientRegistration, [name]: value });
  };
  const clearPatientDetials = () => {
    SetPatientPredictionDetails(PatientPredictionDetails);
  };
  // image handler for patient record
  const handleimagechange = (event) => {
    if (event.target.files && event.target.files[0]) {
      SetFileuploadComplete(event.target.files[0]);
    }
  };
  const formsubmithandler = async (e) => {
    // e.preventDefault();
    setIsLoading(true);
    if (!FileUpload) {
      alert("Please Select the Image for Prediction");
    } else {
      const finalobject = PatientRegistration;
      const json = JSON.stringify(finalobject);
      const formdata = new FormData();
      formdata.append("PatientDetails", json);
      formdata.append("ObservationImage", FileUpload);
      const config = {
        headers: {
          "Content-Type": "multipart/form-data",
        },
      };
      try {
        const result = await axios.post(BASEURL, formdata, config);
        SetPatientPredictionDetails({
          FirstName: result.data.FirstName,
          LastName: result.data.LastName,
          DOB: result.data.DOB,
          Gender: result.data.Gender,
          Phone: result.data.Phone,
          ReportPhoto: result.data.ReportPhoto,
          PredictionResult: result.data.PredictionResult,
        });
        setIsLoading(false);
      } catch (err) {
        alert(`err message is ${err}`);
        setIsLoading(false);
      }
    }
  };
  return (
    <>
      <MDBContainer
        style={{ display: "flex", justifyContent: "space-between" }}
      >
        <MDBRow className="justify-content-center align-items-center m-1">
          <MDBCard>
            <MDBCardBody className="px-4">
              <h3 className="fw-bold mb-4 pb-2 pb-md-0 mb-md-5">
                Patient Details
              </h3>
              <MDBCol>
                <MDBRow>
                  <MDBCol md="6">
                    <MDBInput
                      wrapperClass="mb-4"
                      label="First Name"
                      name="FirstName"
                      size="lg"
                      id="form2"
                      type="text"
                      value={PatientRegistration.FirstName}
                      onChange={handleinput}
                    />
                  </MDBCol>
                </MDBRow>
                <MDBRow>
                  <MDBCol md="6">
                    <MDBInput
                      wrapperClass="mb-4"
                      label="Last Name"
                      name="LastName"
                      size="lg"
                      id="form2"
                      type="text"
                      value={PatientRegistration.LastName}
                      onChange={handleinput}
                    />
                  </MDBCol>
                </MDBRow>
                <MDBRow>
                  <MDBCol md="6">
                    <MDBInput
                      wrapperClass="mb-4"
                      label="DOB"
                      name="DOB"
                      size="lg"
                      id="form3"
                      type="text"
                      value={PatientRegistration.DOB}
                      onChange={handleinput}
                    />
                  </MDBCol>
                </MDBRow>
                <MDBRow>
                  <MDBCol md="6" className="mb-4">
                    <h6 className="fw-bold">Gender: </h6>
                    <MDBRadio
                      id="inlineRadio1"
                      label="Female"
                      name="Gender"
                      value="Female"
                      checked={PatientRegistration.Gender === "Female"}
                      onChange={handleinput}
                      inline
                    />
                    <MDBRadio
                      id="inlineRadio2"
                      name="Gender"
                      label="Male"
                      value="Male"
                      checked={PatientRegistration.Gender === "Male"}
                      onChange={handleinput}
                      inline
                    />
                    <MDBRadio
                      id="inlineRadio3"
                      label="Other"
                      name="Gender"
                      value="Other"
                      checked={PatientRegistration.Gender === "Other"}
                      onChange={handleinput}
                      inline
                    />
                  </MDBCol>
                </MDBRow>
                <MDBRow>
                  <MDBCol md="6">
                    <MDBInput
                      wrapperClass="mb-4"
                      label="Phone"
                      name="Phone"
                      size="lg"
                      id="form5"
                      type="rel"
                      onChange={handleinput}
                      value={PatientRegistration.Phone}
                    />
                  </MDBCol>
                </MDBRow>
                <MDBRow>
                  <MDBCol md="6" className="pe-5">
                    <input
                      type="file"
                      className="form-control"
                      id="customFile"
                      onChange={handleimagechange}
                      accept=".jpg,.jpeg,.png"
                    />
                    <div className="small text-muted mt-2">
                      Upload your X-Ray Image in .jpg, .png, .jpeg{" "}
                    </div>
                  </MDBCol>
                  <MDBRow>
                    <MDBCol md="6">
                      <MDBBtn
                        className="mb-4"
                        size="lg"
                        onClick={formsubmithandler}
                      >
                        Submit
                      </MDBBtn>
                    </MDBCol>
                  </MDBRow>
                </MDBRow>
              </MDBCol>
            </MDBCardBody>
          </MDBCard>
        </MDBRow>
        <MDBRow>
          <SpinnerWidget
            PatientPrediction={PatientPrediction}
            clearPatientDetials={clearPatientDetials}
          />
        </MDBRow>
      </MDBContainer>
    </>
  );
}

export default Test;
