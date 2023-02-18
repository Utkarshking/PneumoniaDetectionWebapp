import React from "react";
import "../App.css";
import {
  MDBBtn,
  MDBContainer,
  MDBCardText,
  MDBRow,
  MDBCol,
  MDBCard,
  MDBCardBody,
  MDBCardImage,
} from "mdb-react-ui-kit";
function SpinnerWidget({PatientPrediction,clearPatientDetials}) {
  const OnClearHandler=()=>{
    console.log("button clicked");
    clearPatientDetials();
  }
  return (
    <div
      className="m-9"
      style={{
        backgroundColor: "#9de2ff",
        width: "600px",
        margin: "0.25rem",
        borderRadius: "10px",
      }}
    >
      <MDBContainer className="m-1">
        <MDBRow className="justify-content-center">
          <MDBCol md="9" lg="7" xl="5" className="m-2 mt-2">
            <MDBCard style={{ borderRadius: "15px", margin: "0.2rem" }}>
              <MDBCardBody className="p-2">
                <div className="d-flex text-black">
                  <div className="flex-shrink-0">
                    <MDBCardImage
                      style={{ width: "11.4rem", borderRadius: "10px" }}
                      src={PatientPrediction.ReportPhoto}
                      alt="Generic placeholder image"
                      fluid
                    />
                    <MDBBtn
                      className="mb-4"
                      size="lg"
                      style={{marginLeft:"30px",marginTop:"30px"}}
                      onClick={OnClearHandler}
                    >
                      Clear
                    </MDBBtn>
                  </div>
                </div>
              </MDBCardBody>
            </MDBCard>
          </MDBCol>
        </MDBRow>
        <MDBCard className="mb-2">
          <MDBCardBody>
            <MDBRow>
              <MDBCol sm="7">
                <MDBCardText>Full Name</MDBCardText>
              </MDBCol>
              <MDBCol sm="9">
                <MDBCardText className="text-muted">{PatientPrediction.FirstName}</MDBCardText>
              </MDBCol>
            </MDBRow>
            <hr />
            <MDBRow>
              <MDBCol sm="7">
                <MDBCardText>DOB</MDBCardText>
              </MDBCol>
              <MDBCol sm="9">
                <MDBCardText className="text-muted">
                  {PatientPrediction.DOB}
                </MDBCardText>
              </MDBCol>
            </MDBRow>
            <hr />
            <MDBRow>
              <MDBCol sm="7">
                <MDBCardText>Gender</MDBCardText>
              </MDBCol>
              <MDBCol sm="9">
                <MDBCardText className="text-muted">{PatientPrediction.Gender}</MDBCardText>
              </MDBCol>
            </MDBRow>
            <hr />
            <MDBRow>
              <MDBCol sm="7">
                <MDBCardText>Phone</MDBCardText>
              </MDBCol>
              <MDBCol sm="9">
                <MDBCardText className="text-muted">{PatientPrediction.Phone}</MDBCardText>
              </MDBCol>
            </MDBRow>
            <hr />
            <MDBRow>
              <MDBRow>
                <MDBCol sm="7">
                  <MDBCardText>Scan Result</MDBCardText>
                </MDBCol>
              </MDBRow>
              <MDBCol sm="9">
                <MDBCardText className="text-muted">
                 {PatientPrediction.PredictionResult}
                </MDBCardText>
              </MDBCol>
            </MDBRow>
          </MDBCardBody>
        </MDBCard>
      </MDBContainer>
    </div>
  );
}

export default SpinnerWidget;
