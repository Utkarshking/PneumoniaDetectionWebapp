import React from 'react'
import 'bootstrap/dist/css/bootstrap.css';
import {Navbar,Container} from "react-bootstrap"
function Header() {
  return (
    <>
    <Navbar bg="primary" expand="sm" variant="dark">
    <Container>
      <Navbar.Brand>Pnuemonia Detection WebApp</Navbar.Brand>
    </Container>
    </Navbar>
    </>
  );
}

export default Header